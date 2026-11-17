// Copyright (C) 2026 Vaibhav
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Real Stockfish binpack streaming pipeline.
//
//   thread A (I/O)      : sequential chunk read, bounded queue
//   thread B (decode)   : sequential binpack parse + rayon feature extraction,
//                         emitted in small blocks so memory stays bounded
//   caller (Python)     : drains blocks with the GIL released
//
// Both producer threads run ahead of the GPU, and every sample records
// (chunk_start, index_in_chunk) which gives exact, cheap resume.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Receiver};
use sfbinpack::read_chunk_into;
use sfbinpack::ChunkReader;
use thiserror::Error;

use crate::features::{active_features, layer_stack_bucket, MAX_ACTIVE};

pub const FEATURE_SLOTS: usize = 2 * MAX_ACTIVE;
const BLOCK_SIZE: usize = 16 * 1024;
const RAW_CAP: usize = 3;
const BLOCK_CAP: usize = 6;
const MAX_CHUNK_SIZE: u64 = 100 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("binpack error: {0}")]
    Binpack(#[from] sfbinpack::CompressedReaderError),
    #[error("invalid binpack chunk near byte {0}")]
    InvalidChunk(u64),
}

#[derive(Clone, Copy)]
pub struct Sample {
    pub features: [i16; FEATURE_SLOTS],
    pub n: [u8; 2],
    pub stm: u8,
    pub bucket: u8,
    pub score: i16,
    pub result: i8,
    pub ply: u16,
    pub chunk_start: u64,
    pub entry_idx: u32,
}

pub struct RawChunk {
    pub start: u64,
    pub bytes: Vec<u8>,
}

pub struct DecodedBlock {
    pub samples: Vec<Sample>,
}

#[inline]
fn build_sample(e: &sfbinpack::TrainingDataEntry, chunk_start: u64, idx: u32) -> Sample {
    let (feat, nw, nb) = active_features(&e.pos);
    let mut features = [0i16; FEATURE_SLOTS];
    for k in 0..FEATURE_SLOTS {
        features[k] = feat[k] as i16;
    }
    Sample {
        features,
        n: [nw, nb],
        stm: if e.pos.side_to_move() == sfbinpack::chess::color::Color::White { 0 } else { 1 },
        bucket: layer_stack_bucket(e.pos.occupied().count()),
        score: e.score,
        result: e.result as i8,
        ply: e.ply,
        chunk_start,
        entry_idx: idx,
    }
}

fn io_thread(
    path: PathBuf,
    start_byte: u64,
    stop: Arc<AtomicBool>,
    tx: crossbeam_channel::Sender<Result<RawChunk, DataError>>,
) {
    let mut file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            let _ = tx.send(Err(DataError::Io(e)));
            return;
        }
    };
    if let Err(e) = file.seek(SeekFrom::Start(start_byte)) {
        let _ = tx.send(Err(DataError::Io(e)));
        return;
    }
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 20);
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let start = match file.stream_position() {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.send(Err(DataError::Io(e)));
                break;
            }
        };
        match read_chunk_into(&mut file, &mut buf) {
            Ok(true) => {
                let ready = std::mem::take(&mut buf);
                if tx.send(Ok(RawChunk { start, bytes: ready })).is_err() {
                    break;
                }
                buf = Vec::with_capacity(1 << 20);
            }
            Ok(false) => break,
            Err(e) => {
                let _ = tx.send(Err(DataError::Binpack(e)));
                break;
            }
        }
    }
}

fn decode_thread(
    stop: Arc<AtomicBool>,
    rx_raw: Receiver<Result<RawChunk, DataError>>,
    tx: crossbeam_channel::Sender<Result<DecodedBlock, DataError>>,
) {
    while !stop.load(Ordering::Relaxed) {
        let raw = match rx_raw.recv() {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                let _ = tx.send(Err(e));
                break;
            }
            Err(_) => break,
        };

        let mut reader = ChunkReader::default();
        let mut block: Vec<(sfbinpack::TrainingDataEntry, u32)> = Vec::with_capacity(BLOCK_SIZE);
        let mut idx: u32 = 0;
        while reader.has_next(&raw.bytes) {
            let entry = reader.next(&raw.bytes);
            block.push((entry, idx));
            idx += 1;
            if block.len() == BLOCK_SIZE {
                if send_block(&tx, &mut block, raw.start).is_err() {
                    return;
                }
            }
        }
        if !block.is_empty() && send_block(&tx, &mut block, raw.start).is_err() {
            return;
        }
    }
}

#[inline]
fn send_block(
    tx: &crossbeam_channel::Sender<Result<DecodedBlock, DataError>>,
    block: &mut Vec<(sfbinpack::TrainingDataEntry, u32)>,
    chunk_start: u64,
) -> Result<(), ()> {
    // Serial feature extraction is parse-bound (~9.5 Mpos/s on this CPU);
    // adding rayon workers only adds contention and memory traffic.
    let drained = std::mem::take(block);
    let samples: Vec<Sample> = drained
        .into_iter()
        .map(|(e, i)| build_sample(&e, chunk_start, i))
        .collect();
    tx.send(Ok(DecodedBlock { samples })).map_err(|_| ())
}

pub struct BinpackPipeline {
    rx: Receiver<Result<DecodedBlock, DataError>>,
    pending: VecDeque<Sample>,
    skip_remaining: u64,
    eof: bool,
    stop: Arc<AtomicBool>,
    handles: Vec<JoinHandle<()>>,
    cursor: (u64, u64),
    pub path: PathBuf,
    pub start_byte: u64,
}

impl BinpackPipeline {
    pub fn open(path: impl AsRef<Path>, start_byte: u64, skip_entries: u64) -> Self {
        let path = path.as_ref().to_path_buf();
        let stop = Arc::new(AtomicBool::new(false));
        let (raw_tx, raw_rx) = bounded(RAW_CAP);
        let (blk_tx, blk_rx) = bounded(BLOCK_CAP);

        let p = path.clone();
        let s1 = stop.clone();
        let h1 = std::thread::Builder::new()
            .name("binpack-io".into())
            .spawn(move || io_thread(p, start_byte, s1, raw_tx))
            .expect("spawn binpack io thread");
        let s2 = stop.clone();
        let h2 = std::thread::Builder::new()
            .name("binpack-decode".into())
            .spawn(move || decode_thread(s2, raw_rx, blk_tx))
            .expect("spawn binpack decode thread");

        Self {
            rx: blk_rx,
            pending: VecDeque::with_capacity(BLOCK_SIZE),
            skip_remaining: skip_entries,
            eof: false,
            stop,
            handles: vec![h1, h2],
            cursor: (start_byte, 0),
            path,
            start_byte,
        }
    }

    fn recv_block(&mut self) -> Result<bool, DataError> {
        match self.rx.recv() {
            Ok(Ok(block)) => {
                let mut samples = block.samples;
                if self.skip_remaining > 0 {
                    let drop_n = self.skip_remaining.min(samples.len() as u64) as usize;
                    samples.drain(..drop_n);
                    self.skip_remaining -= drop_n as u64;
                }
                self.pending.extend(samples);
                Ok(true)
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                self.eof = true;
                Ok(false)
            }
        }
    }

    /// Drain up to `n` samples. Returns fewer only at end of data.
    pub fn next_batch(&mut self, n: usize) -> Result<Vec<Sample>, DataError> {
        while self.pending.len() < n && !self.eof {
            if !self.recv_block()? {
                break;
            }
        }
        let take = n.min(self.pending.len());
        let out: Vec<Sample> = self.pending.drain(..take).collect();
        if let Some(s) = out.last() {
            self.cursor = (s.chunk_start, s.entry_idx as u64 + 1);
        }
        Ok(out)
    }

    pub fn cursor(&self) -> (u64, u64) {
        self.cursor
    }

    pub fn is_eof(&self) -> bool {
        self.eof && self.pending.is_empty()
    }
}

impl Drop for BinpackPipeline {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Replace and drop the receiver first: that unblocks the decode thread
        // if it is stuck in send(), which in turn unblocks the I/O thread.
        let (_, dummy) = bounded::<Result<DecodedBlock, DataError>>(1);
        let real = std::mem::replace(&mut self.rx, dummy);
        drop(real);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

pub fn file_size(path: impl AsRef<Path>) -> Result<u64, DataError> {
    Ok(std::fs::metadata(path)?.len())
}

/// Find the first valid "BINP" chunk header at or after `approx`.
pub fn find_chunk_offset(path: impl AsRef<Path>, approx: u64) -> Result<u64, DataError> {
    let mut f = File::open(path)?;
    let len = f.metadata()?.len();
    let pos = approx.min(len.saturating_sub(8));
    f.seek(SeekFrom::Start(pos))?;
    let mut buf = vec![0u8; 1 << 20];
    let mut base = pos;
    loop {
        let got = f.read(&mut buf)?;
        if got < 8 {
            return Err(DataError::InvalidChunk(approx));
        }
        let mut i = 0usize;
        while i + 8 <= got {
            if &buf[i..i + 4] == b"BINP" {
                let size = u32::from_le_bytes(buf[i + 4..i + 8].try_into().unwrap()) as u64;
                let abs = base + i as u64;
                if size <= MAX_CHUNK_SIZE && abs + 8 + size <= len {
                    return Ok(abs);
                }
            }
            i += 1;
        }
        base += (got - 3) as u64;
        f.seek(SeekFrom::Start(base))?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_real_binpack_if_available() {
        if let Ok(path) = std::env::var("NNUE_TEST_BINPACK") {
            let off = find_chunk_offset(&path, 0).unwrap();
            assert_eq!(off, 0);
            let mut p = BinpackPipeline::open(&path, 0, 0);
            let batch = p.next_batch(1024).unwrap();
            assert!(!batch.is_empty());
            for s in &batch {
                assert!((s.features[0] as usize) < crate::features::INPUT_DIM);
                assert!(s.n[0] as usize <= MAX_ACTIVE);
                assert!(s.bucket <= 7);
            }
        }
    }
}
