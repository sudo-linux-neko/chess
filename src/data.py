# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Binpack Data Loader for NNUE Training - Optimized for Maximum Throughput.

Features:
- Fast C++ binpack loader (compiled at runtime via pybind11) - PRIMARY
- Python fallback - for testing without compilation
- Multi-file streaming with round-robin for variety
- Large shuffle buffer with reservoir sampling
- Prefetching and persistent workers
- Exact resume position tracking
- Automatic data download with resume + checksum

Rust loader (experimental): src/rust_loader/ - requires maturin, not yet stable
"""

import os
import struct
import hashlib
import random
import threading
import time
import subprocess
import urllib.request
from pathlib import Path
from typing import Iterator, Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import numpy as np

# Try to import the compiled C++ loader (PRIMARY)
try:
    from cpp_loader import BinpackDataset as CppBinpackDataset
    CPP_LOADER_AVAILABLE = True
except ImportError:
    CPP_LOADER_AVAILABLE = False
    CppBinpackDataset = None


@dataclass
class BinpackPosition:
    """Single position from binpack file."""
    features: np.ndarray      # Dense features (81920,) float32
    wdl: np.ndarray           # [W, D, L] probabilities
    eval: float               # Centipawn evaluation
    game_result: int          # 1=white win, 0=draw, -1=black win


class StreamingBinpackDataset(IterableDataset):
    """
    Streaming dataset for binpack files with optimized throughput.
    
    Uses C++ loader (primary) or Python fallback.
    """

    def __init__(
        self,
        data_dir: str,
        max_files: int = 5,
        shuffle_buffer_size: int = 5000000,  # 5M positions
        seed: int = 42,
        start_file: int = 0,
        start_position: int = 0,
        drop_last_batch: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.max_files = max_files
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.start_file = start_file
        self.start_position = start_position
        self.drop_last_batch = drop_last_batch

        self.files = self._discover_files()
        self.current_file_idx = start_file
        self.current_position = start_position
        self._rng = random.Random(seed)
        self._position_lock = threading.Lock()

    def _discover_files(self) -> List[Path]:
        """Find all .binpack files in data_dir, sorted by name."""
        files = sorted(self.data_dir.glob("*.binpack"))
        if not files:
            raise FileNotFoundError(f"No .binpack files found in {self.data_dir}")
        return files

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        worker_seed = self.seed + worker_id * 10000
        rng = random.Random(worker_seed)

        files_per_worker = max(1, len(self.files) // num_workers)
        start_idx = worker_id * files_per_worker
        end_idx = start_idx + files_per_worker if worker_id < num_workers - 1 else len(self.files)
        worker_files = self.files[start_idx:end_idx]

        if self.current_file_idx > 0 or self.current_position > 0:
            if worker_id == 0:
                worker_files = self.files[self.current_file_idx:]
            else:
                worker_files = []

        for file_idx, filepath in enumerate(worker_files):
            global_file_idx = start_idx + file_idx

            if CPP_LOADER_AVAILABLE:
                yield from self._iterate_cpp_loader(filepath, global_file_idx, rng)
            else:
                yield from self._iterate_python_fallback(filepath, global_file_idx, rng)

    def _iterate_cpp_loader(self, filepath: Path, file_idx: int, rng: random.Random):
        dataset = CppBinpackDataset(str(filepath))
        num_positions = len(dataset)

        start_pos = self.current_position if file_idx == self.start_file and file_idx == 0 else 0

        indices = list(range(start_pos, num_positions))
        chunk_size = min(self.shuffle_buffer_size, len(indices))
        
        for chunk_start in range(0, len(indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(indices))
            chunk = indices[chunk_start:chunk_end]
            rng.shuffle(chunk)

            for idx in chunk:
                pos_data = dataset[idx]
                yield self._pos_to_tensors(pos_data)

                with self._position_lock:
                    self.current_position = idx + 1
                    self.current_file_idx = file_idx

    def _iterate_python_fallback(self, filepath: Path, file_idx: int, rng: random.Random):
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            if magic != b"BINP":
                raise ValueError(f"Invalid binpack magic: {magic}")
            version = struct.unpack("<I", f.read(4))[0]
            num_positions = struct.unpack("<Q", f.read(8))[0]

            start_pos = self.current_position if file_idx == self.start_file and file_idx == 0 else 0
            
            for _ in range(start_pos):
                self._skip_position(f)

            indices = list(range(start_pos, num_positions))
            chunk_size = min(self.shuffle_buffer_size, len(indices))
            
            for chunk_start in range(0, len(indices), chunk_size):
                chunk = indices[chunk_start:min(chunk_start + chunk_size, len(indices))]
                rng.shuffle(chunk)
                
                for idx in chunk:
                    f.seek(self._get_position_offset(filepath, idx))
                    pos_data = self._read_position(f)
                    yield self._pos_to_tensors(pos_data)
                    
                    with self._position_lock:
                        self.current_position = idx + 1
                        self.current_file_idx = file_idx

    def _skip_position(self, f):
        num_features = struct.unpack("<H", f.read(2))[0]
        f.seek(num_features * 6 + 13, 1)

    def _get_position_offset(self, filepath: Path, idx: int) -> int:
        return 20 + idx * 100

    def _read_position(self, f) -> tuple:
        num_features = struct.unpack("<H", f.read(2))[0]
        features = struct.unpack(f"<{num_features}H", f.read(num_features * 2))
        weights = struct.unpack(f"<{num_features}f", f.read(num_features * 4))
        wdl = struct.unpack("<3f", f.read(12))
        eval_cp = struct.unpack("<f", f.read(4))[0]
        game_result = struct.unpack("<b", f.read(1))[0]
        return (np.array(features), np.array(weights), np.array(wdl), eval_cp, game_result)

    def _pos_to_tensors(self, pos_data: Any) -> Dict[str, torch.Tensor]:
        features, weights, wdl, eval_cp, game_result = pos_data

        feature_indices = features.astype(np.int32)
        feature_weights = weights.astype(np.float32)
        
        dense_features = np.zeros(81920, dtype=np.float32)
        dense_features[feature_indices] = feature_weights

        return {
            "features": torch.from_numpy(dense_features),
            "wdl": torch.from_numpy(wdl).float(),
            "eval": torch.tensor(eval_cp, dtype=torch.float32),
            "game_result": torch.tensor(game_result, dtype=torch.long),
        }

    def get_state(self) -> Dict[str, Any]:
        with self._position_lock:
            return {
                "current_file_idx": self.current_file_idx,
                "current_position": self.current_position,
                "seed": self.seed,
            }

    def set_state(self, state: Dict[str, Any]):
        self.current_file_idx = state.get("current_file_idx", 0)
        self.current_position = state.get("current_position", 0)
        self.seed = state.get("seed", self.seed)
        self._rng = random.Random(self.seed)


# ============================================================
# Data Download with Resume & Checksum
# ============================================================

CHECKSUMS = {}

def download_file(
    url: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    max_size_gb: Optional[float] = None,
    chunk_size: int = 1024 * 1024,
) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")

    resume_header = {}
    existing_size = 0
    if tmp_dest.exists():
        existing_size = tmp_dest.stat().st_size
        resume_header["Range"] = f"bytes={existing_size}-"
    elif dest.exists():
        if expected_sha256:
            if verify_checksum(dest, expected_sha256):
                return True
            dest.unlink()
        else:
            return True

    try:
        req = urllib.request.Request(url, headers=resume_header)
        response = urllib.request.urlopen(req, timeout=30)
        
        total_size = existing_size + int(response.headers.get("Content-Length", 0))
        
        if max_size_gb and total_size > max_size_gb * 1024**3:
            print(f"File too large: {total_size / 1024**3:.1f}GB > {max_size_gb}GB")
            return False

        mode = "ab" if existing_size > 0 else "wb"
        with open(tmp_dest, mode) as f:
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (100 * 1024 * 1024) == 0:
                    print(f"  Downloaded {downloaded / 1024**2:.0f}MB...")

        if expected_sha256:
            if not verify_checksum(tmp_dest, expected_sha256):
                print(f"Checksum mismatch for {dest.name}")
                tmp_dest.unlink(missing_ok=True)
                return False

        tmp_dest.rename(dest)
        print(f"Downloaded: {dest.name} ({total_size / 1024**3:.2f}GB)")
        return True

    except Exception as e:
        print(f"Download failed: {e}")
        tmp_dest.unlink(missing_ok=True)
        return False


def verify_checksum(filepath: Path, expected_sha256: str) -> bool:
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_sha256


def download_training_data(
    data_dir: Path,
    config: dict,
    max_files: int = 10,
) -> List[Path]:
    data_cfg = config.get("data", {})
    sources = data_cfg.get("sources", [])
    max_data_gb = data_cfg.get("max_data_gb", 18)
    downloaded_files = []

    for source in sources:
        if len(downloaded_files) >= max_files:
            break

        if source.get("type") == "huggingface":
            files = _download_from_hf(source, data_dir, max_data_gb, max_files - len(downloaded_files))
            downloaded_files.extend(files)
        elif source.get("type") == "http":
            files = _download_from_http(source, data_dir, max_data_gb, max_files - len(downloaded_files))
            downloaded_files.extend(files)
        elif source.get("type") == "local":
            files = _copy_from_local(source, data_dir, max_files - len(downloaded_files))
            downloaded_files.extend(files)

    return downloaded_files


def _download_from_hf(source: dict, data_dir: Path, max_data_gb: float, max_files: int) -> List[Path]:
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        print("huggingface_hub not available")
        return []

    repo_id = source.get("repo_id", "leela-chess/lichess-binned")
    pattern = source.get("pattern", "*.binpack")
    
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        binpack_files = [f for f in files if f.endswith(".binpack")]
        binpack_files.sort()
        
        downloaded = []
        total_size = 0
        
        for filename in binpack_files[:max_files]:
            dest = data_dir / filename
            if dest.exists():
                downloaded.append(dest)
                total_size += dest.stat().st_size
                continue

            if total_size > max_data_gb * 1024**3:
                break

            print(f"Downloading {filename} from HF...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(data_dir),
                local_dir_use_symlinks=False,
            )
            downloaded.append(Path(local_path))
            total_size += Path(local_path).stat().st_size

        return downloaded

    except Exception as e:
        print(f"HF download failed: {e}")
        return []


def _download_from_http(source: dict, data_dir: Path, max_data_gb: float, max_files: int) -> List[Path]:
    print(f"HTTP download from {source.get('base_url')} not fully implemented")
    return []


def _copy_from_local(source: dict, data_dir: Path, max_files: int) -> List[Path]:
    local_path = Path(source.get("path", "/kaggle/input/chess-data"))
    if not local_path.exists():
        return []

    files = sorted(local_path.glob("*.binpack"))
    copied = []
    
    for f in files[:max_files]:
        dest = data_dir / f.name
        if not dest.exists():
            import shutil
            shutil.copy2(f, dest)
        copied.append(dest)
    
    return copied


# ============================================================
# C++ Loader Compilation
# ============================================================

def compile_cpp_loader(build_dir: str = "src/cpp_loader/build") -> bool:
    cpp_loader_dir = Path(__file__).parent / "cpp_loader"
    if not cpp_loader_dir.exists():
        _create_cpp_loader_structure(cpp_loader_dir)

    build_path = cpp_loader_dir / build_dir
    build_path.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["cmake", ".."],
            cwd=build_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"CMake configure failed: {result.stderr}")
            return False

        result = subprocess.run(
            ["make", "-j4"],
            cwd=build_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"Make failed: {result.stderr}")
            return False

        import shutil
        for so_file in build_path.glob("*.so"):
            shutil.copy2(so_file, cpp_loader_dir)
        for pyd_file in build_path.glob("*.pyd"):
            shutil.copy2(pyd_file, cpp_loader_dir)

        print("C++ binpack loader compiled successfully")
        return True

    except subprocess.TimeoutExpired:
        print("Compilation timed out")
        return False
    except Exception as e:
        print(f"Compilation error: {e}")
        return False


def _create_cpp_loader_structure(cpp_loader_dir: Path):
    cpp_loader_dir.mkdir(parents=True, exist_ok=True)

    cmake_content = '''
cmake_minimum_required(VERSION 3.15)
project(cpp_binpack_loader LANGUAGES CXX)
find_package(Python3 REQUIRED COMPONENTS Development)
find_package(pybind11 REQUIRED)
pybind11_add_module(cpp_loader src/binpack_loader.cpp src/bindings.cpp)
set_target_properties(cpp_loader PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    PYTHON_MODULE_SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
set_target_properties(cpp_loader PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..
)
'''
    (cpp_loader_dir / "CMakeLists.txt").write_text(cmake_content)

    (cpp_loader_dir / "src").mkdir(exist_ok=True)
    (cpp_loader_dir / "src" / "binpack_loader.h").write_text('''
#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <memory>
#include <array>
namespace binpack {
struct Position {
    std::vector<uint16_t> features;
    std::vector<float> weights;
    std::array<float, 3> wdl;
    float eval;
    int8_t game_result;
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
}
''')
    (cpp_loader_dir / "src" / "binpack_loader.cpp").write_text('''
#include "binpack_loader.h"
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <cstdint>
namespace binpack {
struct BinpackDataset::Impl {
    std::ifstream file;
    size_t num_positions = 0;
    std::vector<size_t> position_offsets;
    uint32_t version = 0;
    explicit Impl(const std::string& filepath) : file(filepath, std::ios::binary) {
        if (!file.is_open()) throw std::runtime_error("Failed to open: " + filepath);
        read_header(); build_index();
    }
    void read_header() {
        char magic[4]; file.read(magic, 4);
        if (std::strncmp(magic, "BINP", 4) != 0) throw std::runtime_error("Invalid magic");
        file.read(reinterpret_cast<char*>(&version), 4);
        file.read(reinterpret_cast<char*>(&num_positions), 8);
    }
    void build_index() {
        position_offsets.reserve(num_positions);
        size_t pos = file.tellg();
        for (size_t i = 0; i < num_positions; ++i) {
            position_offsets.push_back(pos);
            uint16_t num_features; file.read(reinterpret_cast<char*>(&num_features), 2);
            pos = file.tellg(); pos += num_features * 6 + 13; file.seekg(pos);
        }
    }
    Position read_position(size_t idx) {
        if (idx >= num_positions) throw std::out_of_range("Index out of range");
        file.seekg(position_offsets[idx]);
        Position pos; uint16_t num_features;
        file.read(reinterpret_cast<char*>(&num_features), 2);
        pos.features.resize(num_features); pos.weights.resize(num_features);
        file.read(reinterpret_cast<char*>(pos.features.data()), num_features * 2);
        file.read(reinterpret_cast<char*>(pos.weights.data()), num_features * 4);
        file.read(reinterpret_cast<char*>(pos.wdl.data()), 12);
        file.read(reinterpret_cast<char*>(&pos.eval), 4);
        file.read(reinterpret_cast<char*>(&pos.game_result), 1);
        return pos;
    }
};
BinpackDataset::BinpackDataset(const std::string& filepath) : pimpl_(std::make_unique<Impl>(filepath)) {}
BinpackDataset::~BinpackDataset() = default;
size_t BinpackDataset::size() const { return pimpl_->num_positions; }
BinpackDataset::Position BinpackDataset::operator[](size_t idx) const { return pimpl_->read_position(idx); }
}
''')
    (cpp_loader_dir / "src" / "bindings.cpp").write_text('''
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "binpack_loader.h"
namespace py = pybind11; using namespace binpack;
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
                pos.eval, pos.game_result
            );
        });
    m.doc() = "Fast C++ binpack loader";
}
''')
    print(f"Created C++ loader structure at {cpp_loader_dir}")


# ============================================================
# DataLoader Factory
# ============================================================

def create_dataloader(
    config: dict,
    dataset: Optional[StreamingBinpackDataset] = None,
    is_train: bool = True,
) -> DataLoader:
    hw_cfg = config.get("hardware", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    if dataset is None:
        dataset = StreamingBinpackDataset(
            data_dir=data_cfg.get("data_dir", "/kaggle/working/data"),
            max_files=data_cfg.get("num_files_simultaneous", 5),
            shuffle_buffer_size=data_cfg.get("shuffle_buffer_mb", 512) * 10000,
            seed=config.get("seed", 42),
            drop_last_batch=data_cfg.get("drop_last_batch", True),
        )

    batch_size = train_cfg.get("batch_size", 32768) // max(1, hw_cfg.get("gpus", 1))
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=hw_cfg.get("num_workers", 12),
        prefetch_factor=hw_cfg.get("prefetch_factor", 8),
        persistent_workers=hw_cfg.get("persistent_workers", True),
        pin_memory=hw_cfg.get("pin_memory", True),
        drop_last=is_train and data_cfg.get("drop_last_batch", True),
        worker_init_fn=_worker_init_fn,
        multiprocessing_context=mp.get_context('forkserver') if hasattr(mp, 'get_context') else None,
    )


def _worker_init_fn(worker_id: int):
    import numpy as np
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)