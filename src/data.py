# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Binpack Data Loader for NNUE Training.

Wraps the fast C++ binpack loader from nnue-pytorch.
Provides streaming, shuffling, multi-worker loading.
"""

import os
import glob
import random
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import numpy as np

# Try to import the compiled C++ loader
try:
    from cpp_loader import BinpackDataset as CppBinpackDataset
    CPP_LOADER_AVAILABLE = True
except ImportError:
    CPP_LOADER_AVAILABLE = False
    CppBinpackDataset = None


@dataclass
class BinpackPosition:
    """Single position from binpack file."""
    features: np.ndarray      # Sparse features (indices)
    weights: np.ndarray       # Feature weights (usually 1.0)
    wdl: np.ndarray           # [W, D, L] probabilities or one-hot
    eval: float               # Centipawn evaluation
    game_result: int          # 1=white win, 0=draw, -1=black win


class BinpackFileIndex:
    """Index for a single binpack file - maps position offset to file position."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.num_positions = 0
        self.position_offsets: List[int] = []
        self._build_index()

    def _build_index(self):
        """Build index by scanning the file."""
        # For now, we'll use the C++ loader which handles this internally
        # This is a fallback for pure Python mode
        pass


class StreamingBinpackDataset(IterableDataset):
    """
    Streaming dataset for binpack files.
    Supports multi-file rotation, shuffling, and resume from position.
    """

    def __init__(
        self,
        data_dir: str,
        max_files: int = 3,
        shuffle_buffer_size: int = 1000000,
        seed: int = 42,
        start_file: int = 0,
        start_position: int = 0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.max_files = max_files
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.start_file = start_file
        self.start_position = start_position

        self.files = self._discover_files()
        self.current_file_idx = start_file
        self.current_position = start_position
        self._rng = random.Random(seed)

        # For resume tracking
        self._position_lock = threading.Lock()

    def _discover_files(self) -> List[Path]:
        """Find all .binpack files in data_dir."""
        files = sorted(self.data_dir.glob("*.binpack"))
        if not files:
            raise FileNotFoundError(f"No .binpack files found in {self.data_dir}")
        return files

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Each worker gets a different seed
        worker_seed = self.seed + worker_id * 1000
        rng = random.Random(worker_seed)

        # Determine which files this worker handles
        files_per_worker = max(1, len(self.files) // num_workers)
        start_idx = worker_id * files_per_worker
        end_idx = start_idx + files_per_worker if worker_id < num_workers - 1 else len(self.files)
        worker_files = self.files[start_idx:end_idx]

        # If resuming, only the first worker handles the resume position
        if self.current_file_idx > 0 or self.current_position > 0:
            if worker_id == 0:
                worker_files = self.files[self.current_file_idx:]
            else:
                worker_files = []

        for file_idx, filepath in enumerate(worker_files):
            global_file_idx = start_idx + file_idx

            # Open file with C++ loader if available
            if CPP_LOADER_AVAILABLE:
                yield from self._iterate_cpp_loader(filepath, global_file_idx, rng)
            else:
                yield from self._iterate_python_fallback(filepath, global_file_idx, rng)

    def _iterate_cpp_loader(self, filepath: Path, file_idx: int, rng: random.Random):
        """Iterate using fast C++ loader."""
        dataset = CppBinpackDataset(str(filepath))

        # Get total positions
        num_positions = len(dataset)

        # Determine start position for this file
        start_pos = self.current_position if file_idx == self.start_file and file_idx == 0 else 0

        # Create indices for shuffling
        indices = list(range(start_pos, num_positions))

        # Shuffle in chunks to avoid memory issues
        chunk_size = min(self.shuffle_buffer_size, len(indices))
        for i in range(0, len(indices), chunk_size):
            chunk = indices[i:i + chunk_size]
            rng.shuffle(chunk)

            for idx in chunk:
                # Get position from C++ loader
                pos_data = dataset[idx]

                # Convert to tensors
                yield self._pos_to_tensors(pos_data)

                # Update resume position
                with self._position_lock:
                    self.current_position = idx + 1
                    self.current_file_idx = file_idx

    def _iterate_python_fallback(self, filepath: Path, file_idx: int, rng: random.Random):
        """Pure Python fallback (slow, for testing only)."""
        # This is a placeholder - real implementation would parse binpack format
        # For now, raise informative error
        raise NotImplementedError(
            "Pure Python binpack loader not implemented. "
            "Please compile the C++ loader: python run.py --compile-only"
        )

    def _pos_to_tensors(self, pos_data: Any) -> Dict[str, torch.Tensor]:
        """Convert C++ loader output to PyTorch tensors."""
        # Expected pos_data format from C++ loader:
        # (features, weights, wdl, eval, game_result)
        features, weights, wdl, eval_cp, game_result = pos_data

        return {
            "features": torch.from_numpy(features).long(),      # Sparse indices
            "weights": torch.from_numpy(weights).float(),       # Feature weights
            "wdl": torch.from_numpy(wdl).float(),               # [3] W/D/L
            "eval": torch.tensor(eval_cp, dtype=torch.float32), # Centipawns
            "game_result": torch.tensor(game_result, dtype=torch.long),
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current dataset state for checkpointing."""
        return {
            "current_file_idx": self.current_file_idx,
            "current_position": self.current_position,
            "seed": self.seed,
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore dataset state from checkpoint."""
        self.current_file_idx = state.get("current_file_idx", 0)
        self.current_position = state.get("current_position", 0)
        self.seed = state.get("seed", self.seed)
        self._rng = random.Random(self.seed)


def create_dataloader(
    config: dict,
    dataset: Optional[StreamingBinpackDataset] = None,
    is_train: bool = True,
) -> DataLoader:
    """Create DataLoader with optimal settings for binpack data."""
    hw_cfg = config.get("hardware", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    if dataset is None:
        dataset = StreamingBinpackDataset(
            data_dir=data_cfg.get("data_dir", "/kaggle/working/data"),
            max_files=data_cfg.get("num_files_simultaneous", 3),
            shuffle_buffer_size=data_cfg.get("shuffle_buffer_mb", 100) * 10000,  # Rough conversion
            seed=config.get("seed", 42),
        )

    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 16384) // max(1, hw_cfg.get("gpus", 1)),
        num_workers=hw_cfg.get("num_workers", 8),
        prefetch_factor=hw_cfg.get("prefetch_factor", 4),
        persistent_workers=hw_cfg.get("persistent_workers", True),
        pin_memory=hw_cfg.get("pin_memory", True),
        drop_last=is_train,
        worker_init_fn=_worker_init_fn,
    )


def _worker_init_fn(worker_id: int):
    """Initialize worker with unique seed."""
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)


def compile_cpp_loader(build_dir: str = "src/cpp_loader/build") -> bool:
    """
    Compile the C++ binpack loader.
    Returns True on success.
    """
    import subprocess
    import sys

    cpp_loader_dir = Path(__file__).parent / "cpp_loader"
    if not cpp_loader_dir.exists():
        # Create minimal C++ loader structure
        _create_cpp_loader_structure(cpp_loader_dir)

    build_path = cpp_loader_dir / build_dir
    build_path.mkdir(parents=True, exist_ok=True)

    try:
        # Configure
        result = subprocess.run(
            ["cmake", ".."],
            cwd=build_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"CMake configure failed: {result.stderr}")
            return False

        # Build
        result = subprocess.run(
            ["make", "-j4"],
            cwd=build_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"Make failed: {result.stderr}")
            return False

        # Copy .so to src/ for import
        import shutil
        for so_file in build_path.glob("*.so"):
            shutil.copy2(so_file, cpp_loader_dir)
        for so_file in build_path.glob("*.pyd"):  # Windows
            shutil.copy2(so_file, cpp_loader_dir)

        print("C++ binpack loader compiled successfully")
        return True

    except subprocess.TimeoutExpired:
        print("Compilation timed out")
        return False
    except Exception as e:
        print(f"Compilation error: {e}")
        return False


def _create_cpp_loader_structure(cpp_loader_dir: Path):
    """Create minimal C++ loader structure for compilation."""
    cpp_loader_dir.mkdir(parents=True, exist_ok=True)

    # CMakeLists.txt
    cmake_content = '''
cmake_minimum_required(VERSION 3.15)
project(cpp_binpack_loader LANGUAGES CXX)

# Python binding with pybind11
find_package(pybind11 REQUIRED)

# Add the loader
pybind11_add_module(cpp_loader src/binpack_loader.cpp src/bindings.cpp)

# Set C++ standard
set_target_properties(cpp_loader PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Output directly to parent directory for easy import
set_target_properties(cpp_loader PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..
)
'''
    (cpp_loader_dir / "CMakeLists.txt").write_text(cmake_content)

    # src/binpack_loader.h
    header_content = '''
#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <memory>

namespace binpack {

struct Position {
    std::vector<uint16_t> features;  // Sparse feature indices
    std::vector<float> weights;      // Feature weights (usually 1.0)
    std::array<float, 3> wdl;        // Win/Draw/Loss probabilities
    float eval;                      // Centipawn evaluation
    int8_t game_result;              // 1=White win, 0=Draw, -1=Black win
};

class BinpackDataset {
public:
    explicit BinpackDataset(const std::string& filepath);
    ~BinpackDataset();

    size_t size() const;
    Position operator[](size_t idx) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace binpack
'''
    (cpp_loader_dir / "src").mkdir(exist_ok=True)
    (cpp_loader_dir / "src" / "binpack_loader.h").write_text(header_content)

    # src/binpack_loader.cpp
    cpp_content = '''
#include "binpack_loader.h"
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cstring>
#include <stdexcept>

namespace binpack {

struct BinpackDataset::Impl {
    std::ifstream file;
    size_t num_positions = 0;
    std::vector<size_t> position_offsets;
    uint32_t version = 0;

    explicit Impl(const std::string& filepath) : file(filepath, std::ios::binary) {
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open binpack file: " + filepath);
        }
        read_header();
        build_index();
    }

    void read_header() {
        // Binpack header: 4 bytes magic, 4 bytes version, 8 bytes num_positions
        char magic[4];
        file.read(magic, 4);
        if (std::strncmp(magic, "BINP", 4) != 0) {
            throw std::runtime_error("Invalid binpack magic number");
        }
        file.read(reinterpret_cast<char*>(&version), 4);
        file.read(reinterpret_cast<char*>(&num_positions), 8);
    }

    void build_index() {
        position_offsets.reserve(num_positions);
        size_t pos = file.tellg();
        for (size_t i = 0; i < num_positions; ++i) {
            position_offsets.push_back(pos);
            // Skip position data: 2 bytes num_features, then features (2 bytes each),
            // weights (4 bytes each), wdl (3*4), eval (4), game_result (1)
            uint16_t num_features;
            file.read(reinterpret_cast<char*>(&num_features), 2);
            pos = file.tellg();
            pos += num_features * (2 + 4); // features + weights
            pos += 3 * 4; // wdl
            pos += 4;     // eval
            pos += 1;     // game_result
            file.seekg(pos);
        }
    }

    Position read_position(size_t idx) {
        if (idx >= num_positions) {
            throw std::out_of_range("Position index out of range");
        }
        file.seekg(position_offsets[idx]);
        Position pos;
        uint16_t num_features;
        file.read(reinterpret_cast<char*>(&num_features), 2);
        pos.features.resize(num_features);
        pos.weights.resize(num_features);
        file.read(reinterpret_cast<char*>(pos.features.data()), num_features * 2);
        file.read(reinterpret_cast<char*>(pos.weights.data()), num_features * 4);
        file.read(reinterpret_cast<char*>(pos.wdl.data()), 3 * 4);
        file.read(reinterpret_cast<char*>(&pos.eval), 4);
        file.read(reinterpret_cast<char*>(&pos.game_result), 1);
        return pos;
    }
};

BinpackDataset::BinpackDataset(const std::string& filepath)
    : pimpl_(std::make_unique<Impl>(filepath)) {}

BinpackDataset::~BinpackDataset() = default;

size_t BinpackDataset::size() const {
    return pimpl_->num_positions;
}

BinpackDataset::Position BinpackDataset::operator[](size_t idx) const {
    return pimpl_->read_position(idx);
}

} // namespace binpack
'''
    (cpp_loader_dir / "src" / "binpack_loader.cpp").write_text(cpp_content)

    # src/bindings.cpp
    bindings_content = '''
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "binpack_loader.h"

namespace py = pybind11;
using namespace binpack;

PYBIND11_MODULE(cpp_loader, m) {
    py::class_<BinpackDataset>(m, "BinpackDataset")
        .def(py::init<const std::string&>())
        .def("__len__", &BinpackDataset::size)
        .def("__getitem__", [](const BinpackDataset& self, size_t idx) {
            auto pos = self[idx];
            return py::make_tuple(
                py::array_t<uint16_t>(pos.features.size(), pos.features.data()),
                py::array_t<float>(pos.weights.size(), pos.weights.data()),
                py::array_t<float>(3, pos.wdl.data()),
                pos.eval,
                pos.game_result
            );
        });

    m.doc() = "Fast C++ binpack loader for NNUE training";
}
'''
    (cpp_loader_dir / "src" / "bindings.cpp").write_text(bindings_content)

    print(f"Created C++ loader structure at {cpp_loader_dir}")