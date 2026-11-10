// Copyright (C) 2026 Vaibhav
// SPDX-License-Identifier: GPL-3.0-or-later

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyArray1, PyArray2, IntoPyArray};
use ndarray::{Array1, Array2, Axis, ShapeBuilder};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use byteorder::{LittleEndian, ReadBytesExt};
use thiserror::Error;

const MAGIC: &[u8; 4] = b"BINP";
const VERSION: u32 = 1;
const FEATURES_PER_BUCKET: usize = 1280;
const NUM_BUCKETS: usize = 64;
const INPUT_DIM: usize = NUM_BUCKETS * FEATURES_PER_BUCKET;

#[derive(Error, Debug)]
pub enum BinpackError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid magic number: expected BINP, got {0:?}")]
    InvalidMagic([u8; 4]),
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("Index out of bounds: {0}")]
    OutOfBounds(usize),
    #[error("Invalid feature index: {0}")]
    InvalidFeatureIndex(usize),
}

impl From<BinpackError> for PyErr {
    fn from(err: BinpackError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct Position {
    pub features: Array1<f32>,
    pub wdl: [f32; 3],
    pub eval: f32,
    pub game_result: i8,
}

pub struct BinpackDataset {
    file: File,
    num_positions: u64,
    position_offsets: Vec<u64>,
}

impl BinpackDataset {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, BinpackError> {
        let mut file = File::open(path)?;
        let mut reader = BufReader::new(&mut file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != *MAGIC {
            return Err(BinpackError::InvalidMagic(magic));
        }

        let version = reader.read_u32::<LittleEndian>()?;
        if version != VERSION {
            return Err(BinpackError::UnsupportedVersion(version));
        }

        let num_positions = reader.read_u64::<LittleEndian>()?;

        let mut position_offsets = Vec::with_capacity(num_positions as usize);
        let mut current_pos = reader.stream_position()?;

        for _ in 0..num_positions {
            position_offsets.push(current_pos);

            let num_features = reader.read_u16::<LittleEndian>()? as u64;
            current_pos += 2 + num_features * (2 + 4) + 12 + 4 + 1;
            reader.seek(SeekFrom::Start(current_pos))?;
        }

        Ok(Self {
            file,
            num_positions,
            position_offsets,
        })
    }

    pub fn len(&self) -> u64 {
        self.num_positions
    }

    pub fn get(&mut self, index: u64) -> Result<Position, BinpackError> {
        if index >= self.num_positions {
            return Err(BinpackError::OutOfBounds(index as usize));
        }

        let offset = self.position_offsets[index as usize];
        self.file.seek(SeekFrom::Start(offset))?;

        let mut reader = BufReader::new(&mut self.file);

        let num_features = reader.read_u16::<LittleEndian>()? as usize;
        
        let mut feature_indices = vec![0u16; num_features];
        let mut feature_weights = vec![0.0f32; num_features];
        
        for i in 0..num_features {
            feature_indices[i] = reader.read_u16::<LittleEndian>()?;
        }
        for i in 0..num_features {
            feature_weights[i] = reader.read_f32::<LittleEndian>()?;
        }

        let mut wdl = [0.0f32; 3];
        for i in 0..3 {
            wdl[i] = reader.read_f32::<LittleEndian>()?;
        }

        let eval = reader.read_f32::<LittleEndian>()?;
        let game_result = reader.read_i8()?;

        let mut features = Array1::zeros(INPUT_DIM);
        for (idx, &weight) in feature_indices.iter().zip(feature_weights.iter()) {
            let idx = idx as usize;
            if idx < INPUT_DIM {
                features[idx] = weight;
            }
        }

        Ok(Position {
            features,
            wdl,
            eval,
            game_result,
        })
    }
}

#[pyclass]
pub struct PyBinpackDataset {
    inner: BinpackDataset,
}

#[pymethods]
impl PyBinpackDataset {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = BinpackDataset::open(path)?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> u64 {
        self.inner.len()
    }

    fn __getitem__(&mut self, index: usize) -> PyResult<PyObject> {
        let pos = self.inner.get(index as u64)?;
        Python::with_gil(|py| {
            let features = PyArray1::from_vec(py, pos.features.to_vec());
            let wdl = PyArray1::from_vec(py, pos.wdl.to_vec());
            let eval = pos.eval;
            let game_result = pos.game_result;

            let features_obj = features.to_object(py);
            let wdl_obj = wdl.to_object(py);
            let eval_obj = eval.into_py(py);
            let game_result_obj = game_result.into_py(py);

            Ok(PyTuple::new(py, &[
                features_obj,
                wdl_obj,
                eval_obj,
                game_result_obj,
            ]).into_py(py))
        })
    }

    fn get_batch(&mut self, indices: Vec<usize>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let batch_size = indices.len();
            let mut features_batch = Array2::zeros((batch_size, INPUT_DIM));
            let mut wdl_batch = Array2::zeros((batch_size, 3));
            let mut eval_batch = Array1::zeros(batch_size);
            let mut result_batch = Array1::zeros(batch_size);

            for (i, &idx) in indices.iter().enumerate() {
                let pos = self.inner.get(idx as u64)?;
                features_batch.row_mut(i).assign(&pos.features);
                wdl_batch.row_mut(i).assign(&Array1::from_vec(pos.wdl.to_vec()));
                eval_batch[i] = pos.eval;
                result_batch[i] = pos.game_result as f32;
            }

            let features_arr = PyArray2::from_array(py, &features_batch);
            let wdl_arr = PyArray2::from_array(py, &wdl_batch);
            let eval_arr = PyArray1::from_vec(py, eval_batch.to_vec());
            let result_arr = PyArray1::from_vec(py, result_batch.to_vec());

            let features_obj = features_arr.to_object(py);
            let wdl_obj = wdl_arr.to_object(py);
            let eval_obj = eval_arr.to_object(py);
            let result_obj = result_arr.to_object(py);

            Ok(PyTuple::new(py, &[
                features_obj,
                wdl_obj,
                eval_obj,
                result_obj,
            ]).into_py(py))
        })
    }
}

#[pymodule]
fn binpack_loader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBinpackDataset>()?;
    Ok(())
}