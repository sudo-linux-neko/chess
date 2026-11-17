// Copyright (C) 2026 Vaibhav
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stockfish 16.1 "Small" network (.nnue) writer and exact integer
// re-implementation of Stockfish's inference, used to verify exports.
//
// Layout is the real sf_16.1 format:
//   header(version, hash, description)
//   FT hash
//   LEB128 blocks: bias[128], weight[22528*128], psqt[22528*8]
//   8 layer stacks, each: fc hash, l1 bias[16], l1 weight[16*128] i8,
//                         l2 bias[32], l2 weight[32*32] i8,
//                         out bias[1], out weight[1*32] i8
//
// FC weights are written in logical [out][padded_in] order. Stockfish's
// get_weight_index_scrambled() turns that into its SIMD layout.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::features::INPUT_DIM;

pub const L1: usize = 128;
pub const L2: usize = 15;
pub const L3: usize = 32;
pub const LAYER_STACKS: usize = 8;
pub const PSQT_BUCKETS: usize = 8;
pub const VERSION: u32 = 0x7AF3_2F20;
pub const FEATURE_HASH: u32 = 0x7f23_4cb8;

const LEB_MAGIC: &[u8] = b"COMPRESSED_LEB128";

fn pyerr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Signed LEB128, exactly like Stockfish's write_leb_128.
#[inline]
fn leb128_push(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if (byte & 0x40) == 0 {
            if v == 0 {
                out.push(byte);
                break;
            }
        } else if v == -1 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

fn write_leb_block(out: &mut Vec<u8>, values: impl Iterator<Item = i64>) {
    // Encode into a temporary buffer to know the length first.
    let mut payload: Vec<u8> = Vec::new();
    for v in values {
        leb128_push(&mut payload, v);
    }
    out.extend_from_slice(LEB_MAGIC);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
}

fn fc_hash() -> u32 {
    let mut prev: u32 = 0xEC42_E90D ^ ((L1 as u32) * 2);
    for (out_per_bucket, relu) in [(L2 + 1, true), (L3, true), (1, false)] {
        let mut h: u32 = 0xCC03_DAE4;
        h = h.wrapping_add(out_per_bucket as u32);
        h ^= prev >> 1;
        h ^= prev << 31;
        if relu {
            h = h.wrapping_add(0x538D_24C7);
        }
        prev = h;
    }
    prev
}

pub fn ft_hash() -> u32 {
    FEATURE_HASH ^ ((L1 as u32) * 2)
}

fn check_2d<T: numpy::Element>(arr: &PyReadonlyArray2<'_, T>, rows: usize, cols: usize, name: &str) -> PyResult<()> {
    let s = arr.as_array().shape().to_owned();
    if s != [rows, cols] {
        return Err(pyerr(format!(
            "{name}: expected shape ({rows}, {cols}), got ({}, {})",
            s[0], s[1]
        )));
    }
    Ok(())
}

fn check_3d<T: numpy::Element>(
    arr: &PyReadonlyArray3<'_, T>,
    d0: usize,
    d1: usize,
    d2: usize,
    name: &str,
) -> PyResult<()> {
    let s = arr.as_array().shape().to_owned();
    if s != [d0, d1, d2] {
        return Err(pyerr(format!(
            "{name}: expected shape ({d0}, {d1}, {d2}), got ({}, {}, {})",
            s[0], s[1], s[2]
        )));
    }
    Ok(())
}

/// Serialize a quantized SF16.1 Small network.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn write_nnue(
    path: String,
    description: String,
    ft_bias: PyReadonlyArray1<'_, i16>,
    ft_weight: PyReadonlyArray2<'_, i16>,
    psqt: PyReadonlyArray2<'_, i32>,
    l1_bias: PyReadonlyArray2<'_, i32>,
    l1_weight: PyReadonlyArray3<'_, i8>,
    l2_bias: PyReadonlyArray2<'_, i32>,
    l2_weight: PyReadonlyArray3<'_, i8>,
    out_bias: PyReadonlyArray2<'_, i32>,
    out_weight: PyReadonlyArray3<'_, i8>,
) -> PyResult<()> {
    let fb_shape = ft_bias.as_array().shape().to_owned();
    if fb_shape != [L1] {
        return Err(pyerr(format!(
            "ft_bias: expected ({L1},), got ({},)",
            fb_shape[0]
        )));
    }
    check_2d(&ft_weight, INPUT_DIM, L1, "ft_weight")?;
    check_2d(&psqt, INPUT_DIM, PSQT_BUCKETS, "psqt")?;
    check_2d(&l1_bias, LAYER_STACKS, L2 + 1, "l1_bias")?;
    check_3d(&l1_weight, LAYER_STACKS, L2 + 1, L1, "l1_weight")?;
    check_2d(&l2_bias, LAYER_STACKS, L3, "l2_bias")?;
    check_3d(&l2_weight, LAYER_STACKS, L3, L3, "l2_weight")?;
    check_2d(&out_bias, LAYER_STACKS, 1, "out_bias")?;
    check_3d(&out_weight, LAYER_STACKS, 1, L3, "out_weight")?;

    let fb = ft_bias.as_slice()?;
    let fw = ft_weight.as_slice()?;
    let pq = psqt.as_slice()?;
    let b1 = l1_bias.as_slice()?;
    let w1 = l1_weight.as_slice()?;
    let b2 = l2_bias.as_slice()?;
    let w2 = l2_weight.as_slice()?;
    let bo = out_bias.as_slice()?;
    let wo = out_weight.as_slice()?;

    let fch = fc_hash();
    let fth = ft_hash();
    let desc = description.as_bytes();

    let mut out: Vec<u8> = Vec::with_capacity(16 * 1024 * 1024);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(fch ^ fth).to_le_bytes());
    out.extend_from_slice(&(desc.len() as u32).to_le_bytes());
    out.extend_from_slice(desc);
    out.extend_from_slice(&fth.to_le_bytes());

    // FT bias
    write_leb_block(&mut out, fb.iter().map(|&v| v as i64));
    // FT weights, feature-major
    write_leb_block(&mut out, fw.iter().map(|&v| v as i64));
    // PSQT weights, feature-major
    write_leb_block(&mut out, pq.iter().map(|&v| v as i64));

    let l1_stride = (L2 + 1) * L1;
    let l2_stride = L3 * L3;
    let out_stride = L3;
    for s in 0..LAYER_STACKS {
        out.extend_from_slice(&fch.to_le_bytes());
        let b1s = &b1[s * (L2 + 1)..(s + 1) * (L2 + 1)];
        for &v in b1s {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let w1s = &w1[s * l1_stride..(s + 1) * l1_stride];
        out.extend_from_slice(as_u8_slice(w1s));

        let b2s = &b2[s * L3..(s + 1) * L3];
        for &v in b2s {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let w2s = &w2[s * l2_stride..(s + 1) * l2_stride];
        out.extend_from_slice(as_u8_slice(w2s));

        out.extend_from_slice(&bo[s].to_le_bytes());
        let wos = &wo[s * out_stride..(s + 1) * out_stride];
        out.extend_from_slice(as_u8_slice(wos));
    }

    let mut f = BufWriter::new(File::create(Path::new(&path)).map_err(|e| pyerr(e.to_string()))?);
    f.write_all(&out).map_err(|e| pyerr(e.to_string()))?;
    f.flush().map_err(|e| pyerr(e.to_string()))?;
    Ok(())
}

#[inline]
fn as_u8_slice(v: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

// ---------------------------------------------------------------------------
// Parser + exact integer inference (verification only)
// ---------------------------------------------------------------------------

struct Net {
    ft_bias: Vec<i16>,
    ft_weight: Vec<i16>,
    psqt: Vec<i32>,
    l1_bias: Vec<i32>,
    l1_weight: Vec<i8>,
    l2_bias: Vec<i32>,
    l2_weight: Vec<i8>,
    out_bias: Vec<i32>,
    out_weight: Vec<i8>,
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32, String> {
        if self.p + 4 > self.b.len() {
            return Err("unexpected EOF".into());
        }
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        Ok(v)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p + n > self.b.len() {
            return Err("unexpected EOF".into());
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn i32s(&mut self, n: usize) -> Result<Vec<i32>, String> {
        let s = self.bytes(n * 4)?;
        Ok(s.chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
    fn i8s(&mut self, n: usize) -> Result<Vec<i8>, String> {
        let s = self.bytes(n)?;
        Ok(s.iter().map(|&v| v as i8).collect())
    }
    fn leb_magic(&mut self) -> Result<(), String> {
        let m = self.bytes(LEB_MAGIC.len())?;
        if m != LEB_MAGIC {
            return Err("missing COMPRESSED_LEB128 magic".into());
        }
        Ok(())
    }
    fn leb_block(&mut self, count: usize) -> Result<Vec<i64>, String> {
        self.leb_magic()?;
        let n = self.u32()? as usize;
        let data = self.bytes(n)?;
        let mut out = Vec::with_capacity(count);
        let mut i = 0usize;
        for _ in 0..count {
            let mut result: i64 = 0;
            let mut shift: u32 = 0;
            loop {
                if i >= data.len() || shift >= 64 {
                    return Err("bad LEB128 stream".into());
                }
                let byte = data[i];
                i += 1;
                result |= ((byte & 0x7f) as i64) << shift;
                shift += 7;
                if byte & 0x80 == 0 {
                    if shift < 64 && byte & 0x40 != 0 {
                        result |= -1i64 << shift;
                    }
                    break;
                }
            }
            out.push(result);
        }
        Ok(out)
    }
}

fn parse_net(path: &str) -> Result<Net, String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    let mut r = Reader { b: &data, p: 0 };
    let version = r.u32()?;
    if version != VERSION {
        return Err(format!("bad version {version:#x}, expected {VERSION:#x}"));
    }
    let net_hash = r.u32()?;
    let desc_len = r.u32()? as usize;
    let _desc = r.bytes(desc_len)?;
    let fth = r.u32()?;
    if fth != ft_hash() {
        return Err(format!("bad FT hash {fth:#x}, expected {:#x}", ft_hash()));
    }
    if net_hash != fc_hash() ^ ft_hash() {
        return Err(format!(
            "bad net hash {net_hash:#x}, expected {:#x}",
            fc_hash() ^ ft_hash()
        ));
    }

    let bias_i64 = r.leb_block(L1)?;
    let weight_i64 = r.leb_block(INPUT_DIM * L1)?;
    let psqt_i64 = r.leb_block(INPUT_DIM * PSQT_BUCKETS)?;
    let ft_bias: Vec<i16> = bias_i64.iter().map(|&v| v as i16).collect();
    let ft_weight: Vec<i16> = weight_i64.iter().map(|&v| v as i16).collect();
    let psqt: Vec<i32> = psqt_i64.iter().map(|&v| v as i32).collect();

    let mut l1_bias = Vec::with_capacity(LAYER_STACKS * (L2 + 1));
    let mut l1_weight = Vec::with_capacity(LAYER_STACKS * (L2 + 1) * L1);
    let mut l2_bias = Vec::with_capacity(LAYER_STACKS * L3);
    let mut l2_weight = Vec::with_capacity(LAYER_STACKS * L3 * L3);
    let mut out_bias = Vec::with_capacity(LAYER_STACKS);
    let mut out_weight = Vec::with_capacity(LAYER_STACKS * L3);
    let fch = fc_hash();
    for _ in 0..LAYER_STACKS {
        if r.u32()? != fch {
            return Err("bad FC hash".into());
        }
        l1_bias.extend(r.i32s(L2 + 1)?);
        l1_weight.extend(r.i8s((L2 + 1) * L1)?);
        l2_bias.extend(r.i32s(L3)?);
        l2_weight.extend(r.i8s(L3 * L3)?);
        out_bias.extend(r.i32s(1)?);
        out_weight.extend(r.i8s(L3)?);
    }
    if r.p != data.len() {
        return Err("trailing bytes after network".into());
    }

    Ok(Net {
        ft_bias,
        ft_weight,
        psqt,
        l1_bias,
        l1_weight,
        l2_bias,
        l2_weight,
        out_bias,
        out_weight,
    })
}

#[inline]
fn clamp_u8(v: i32) -> i32 {
    v.clamp(0, 127)
}

impl Net {
    fn eval_one(&self, feats: &[i16], stm: u8, out: &mut Vec<i32>) -> Result<(), String> {
        let mut acc = [[0i32; L1]; 2];
        let mut psqt_acc = [[0i32; PSQT_BUCKETS]; 2];
        let mut n = [0usize; 2];
        let mut piece_count = 0usize;
        for (k, &f) in feats.iter().enumerate() {
            let p = k / 32;
            if (f as usize) >= INPUT_DIM {
                continue;
            }
            let idx = f as usize;
            n[p] += 1;
            if p == 0 {
                piece_count += 1;
            }
            let row = &self.ft_weight[idx * L1..(idx + 1) * L1];
            for j in 0..L1 {
                acc[p][j] += row[j] as i32;
            }
            let prow = &self.psqt[idx * PSQT_BUCKETS..(idx + 1) * PSQT_BUCKETS];
            for b in 0..PSQT_BUCKETS {
                psqt_acc[p][b] += prow[b];
            }
        }
        let mut bucket = piece_count.saturating_sub(1) / 4;
        if bucket >= LAYER_STACKS {
            bucket = LAYER_STACKS - 1;
        }
        for p in 0..2 {
            for j in 0..L1 {
                acc[p][j] += self.ft_bias[j] as i32;
                if acc[p][j] < i16::MIN as i32 || acc[p][j] > i16::MAX as i32 {
                    return Err(format!("accumulator overflow: {}", acc[p][j]));
                }
            }
        }

        let persp = [stm as usize, 1 - stm as usize];
        let half = L1 / 2;
        let mut tf = [0u8; L1];
        for p in 0..2 {
            let a = &acc[persp[p]];
            for j in 0..half {
                let lo = clamp_u8(a[j]);
                let hi = clamp_u8(a[j + half]);
                tf[p * half + j] = ((lo * hi) >> 7) as u8;
            }
        }

        let psqt = (psqt_acc[persp[0]][bucket] - psqt_acc[persp[1]][bucket]) / 2;

        let base1 = bucket * (L2 + 1);
        let mut fc0 = [0i32; L2 + 1];
        for o in 0..=L2 {
            let row = &self.l1_weight[(base1 + o) * L1..(base1 + o + 1) * L1];
            let mut s = self.l1_bias[base1 + o];
            for i in 0..L1 {
                s += row[i] as i32 * tf[i] as i32;
            }
            fc0[o] = s;
        }

        let mut in1 = [0u8; 32];
        for j in 0..L2 {
            in1[j] = (((fc0[j] as i64 * fc0[j] as i64) >> 19).min(127)) as u8;
            in1[L2 + j] = clamp_u8(fc0[j] >> 6) as u8;
        }

        let base2 = bucket * L3;
        let mut fc1 = [0i32; L3];
        for o in 0..L3 {
            let row = &self.l2_weight[(base2 + o) * L3..(base2 + o + 1) * L3];
            let mut s = self.l2_bias[base2 + o];
            for i in 0..L3 {
                s += row[i] as i32 * in1[i] as i32;
            }
            fc1[o] = s;
        }
        let mut ac1 = [0u8; L3];
        for i in 0..L3 {
            ac1[i] = clamp_u8(fc1[i] >> 6) as u8;
        }

        let mut fc2 = self.out_bias[bucket];
        for i in 0..L3 {
            fc2 += self.out_weight[bucket * L3 + i] as i32 * ac1[i] as i32;
        }
        let fwd = (fc0[L2] as i64 * 9600 / 8128) as i32;
        out.push((psqt + fc2 + fwd) / 16);
        Ok(())
    }
}

/// Exact Stockfish-side integer evaluation of an exported net.
#[pyfunction]
pub fn eval_nnue(
    path: String,
    features: PyReadonlyArray2<'_, i16>,
    stm: PyReadonlyArray1<'_, u8>,
) -> PyResult<Vec<i32>> {
    let net = parse_net(&path).map_err(pyerr)?;
    let f = features.as_array();
    let s = stm.as_slice()?;
    if f.shape()[0] != s.len() || f.shape()[1] != 2 * crate::features::MAX_ACTIVE {
        return Err(pyerr("features must be (n, 64) and match stm length"));
    }
    let mut out = Vec::with_capacity(s.len());
    for i in 0..s.len() {
        let row: Vec<i16> = (0..64).map(|k| f[[i, k]]).collect();
        net.eval_one(&row, s[i], &mut out).map_err(pyerr)?;
    }
    Ok(out)
}

#[pyfunction]
pub fn nnue_info(path: String) -> PyResult<(u32, u32, u32, u32)> {
    let data = std::fs::read(&path).map_err(|e| pyerr(e.to_string()))?;
    let mut r = Reader { b: &data, p: 0 };
    let version = r.u32().map_err(pyerr)?;
    let net_hash = r.u32().map_err(pyerr)?;
    let desc_len = r.u32().map_err(pyerr)? as usize;
    let _ = r.bytes(desc_len).map_err(pyerr)?;
    let ft = r.u32().map_err(pyerr)?;
    Ok((version, net_hash, ft, desc_len as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leb_roundtrip() {
        let vals = [0i64, 1, -1, 63, 64, -64, -65, 127, 128, -128, i16::MIN as i64, i16::MAX as i64];
        let mut enc = Vec::new();
        for &v in &vals {
            leb128_push(&mut enc, v);
        }
        let mut i = 0;
        for &v in &vals {
            let mut result: i64 = 0;
            let mut shift = 0;
            loop {
                let byte = enc[i];
                i += 1;
                result |= ((byte & 0x7f) as i64) << shift;
                shift += 7;
                if byte & 0x80 == 0 {
                    if shift < 64 && byte & 0x40 != 0 {
                        result |= -1i64 << shift;
                    }
                    break;
                }
            }
            assert_eq!(result, v);
        }
    }

    #[test]
    fn hashes_are_sf161() {
        assert_eq!(VERSION, 0x7AF32F20);
        assert_eq!(ft_hash(), 0x7f234cb8 ^ 0x100);
    }
}
