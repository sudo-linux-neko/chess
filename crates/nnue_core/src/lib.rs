// Copyright (C) 2026 Vaibhav
// SPDX-License-Identifier: GPL-3.0-or-later

//! Rust core for the chess NNUE trainer.
//!
//! * real Stockfish binpack streaming (chunk-parallel, GIL-free)
//! * exact Stockfish 16.1 HalfKAv2_hm feature indices
//! * resumable cursors at chunk/entry granularity

use ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod dataset;
pub mod features;
pub mod nnue_file;

use dataset::{BinpackPipeline, DataError};
use features::{MAX_ACTIVE, INPUT_DIM};

fn to_pyerr(e: DataError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pyclass(module = "nnue_core")]
struct BinpackReader {
    inner: BinpackPipeline,
}

#[pymethods]
impl BinpackReader {
    #[new]
    #[pyo3(signature = (path, start_byte = 0, skip_entries = 0))]
    fn new(path: String, start_byte: u64, skip_entries: u64) -> Self {
        Self {
            inner: BinpackPipeline::open(path, start_byte, skip_entries),
        }
    }

    /// Drain up to `n` samples. Returns a dict of numpy arrays plus `count`.
    /// All computation runs without the GIL.
    fn next_batch(&mut self, py: Python<'_>, n: usize) -> PyResult<Py<PyDict>> {
        let batch = py.detach(|| self.inner.next_batch(n)).map_err(to_pyerr)?;
        let m = batch.len();

        let mut features: Vec<i16> = Vec::with_capacity(m * 2 * MAX_ACTIVE);
        let mut n_feat: Vec<u8> = Vec::with_capacity(m * 2);
        let mut stm: Vec<u8> = Vec::with_capacity(m);
        let mut bucket: Vec<u8> = Vec::with_capacity(m);
        let mut score: Vec<i16> = Vec::with_capacity(m);
        let mut result: Vec<i8> = Vec::with_capacity(m);
        let mut ply: Vec<u16> = Vec::with_capacity(m);
        for s in &batch {
            features.extend_from_slice(&s.features);
            n_feat.extend_from_slice(&s.n);
            stm.push(s.stm);
            bucket.push(s.bucket);
            score.push(s.score);
            result.push(s.result);
            ply.push(s.ply);
        }

        let d = PyDict::new(py);
        d.set_item(
            "features",
            Array2::from_shape_vec((m, 2 * MAX_ACTIVE), features)
                .unwrap()
                .into_pyarray(py),
        )?;
        d.set_item(
            "n_feat",
            Array2::from_shape_vec((m, 2), n_feat).unwrap().into_pyarray(py),
        )?;
        d.set_item("stm", stm.into_pyarray(py))?;
        d.set_item("bucket", bucket.into_pyarray(py))?;
        d.set_item("score", score.into_pyarray(py))?;
        d.set_item("result", result.into_pyarray(py))?;
        d.set_item("ply", ply.into_pyarray(py))?;
        d.set_item("count", m)?;
        d.set_item("chunk_start", self.inner.cursor().0)?;
        d.set_item("entry_idx", self.inner.cursor().1)?;
        d.set_item("eof", self.inner.is_eof())?;
        Ok(d.unbind())
    }

    /// `(chunk_start_byte, next_entry_index_within_chunk)` for exact resume.
    fn cursor(&self) -> (u64, u64) {
        self.inner.cursor()
    }

    fn eof(&self) -> bool {
        self.inner.is_eof()
    }

    fn start_byte(&self) -> u64 {
        self.inner.start_byte
    }
}

#[pyfunction]
fn find_chunk_offset(path: String, approx: u64) -> PyResult<u64> {
    dataset::find_chunk_offset(path, approx).map_err(to_pyerr)
}

#[pyfunction]
fn file_size(path: String) -> PyResult<u64> {
    dataset::file_size(path).map_err(to_pyerr)
}

/// Debug/test helper: padded HalfKAv2_hm features for a FEN.
/// Returns `(features[64], n_white, n_black, layer_stack_bucket)`.
#[pyfunction]
fn fen_features(fen: String) -> PyResult<(Vec<i16>, u8, u8, u8)> {
    use sfbinpack::chess::position::Position;
    let pos = Position::from_fen(&fen).map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
    let (feat, nw, nb) = features::active_features(&pos);
    let out: Vec<i16> = feat.iter().map(|&v| v as i16).collect();
    let bucket = features::layer_stack_bucket(pos.occupied().count());
    Ok((out, nw, nb, bucket))
}

#[pymodule]
fn nnue_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("INPUT_DIM", INPUT_DIM)?;
    m.add("MAX_ACTIVE", MAX_ACTIVE)?;
    m.add("NUM_BUCKETS", features::NUM_BUCKETS)?;
    m.add("PAD_INDEX", features::PAD_INDEX)?;
    m.add_class::<BinpackReader>()?;
    m.add_function(wrap_pyfunction!(find_chunk_offset, m)?)?;
    m.add_function(wrap_pyfunction!(file_size, m)?)?;
    m.add_function(wrap_pyfunction!(fen_features, m)?)?;
    m.add_function(wrap_pyfunction!(nnue_file::write_nnue, m)?)?;
    m.add_function(wrap_pyfunction!(nnue_file::eval_nnue, m)?)?;
    m.add_function(wrap_pyfunction!(nnue_file::nnue_info, m)?)?;
    Ok(())
}
