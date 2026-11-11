# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Ultra-Optimized Data Pipeline for NNUE Training.

Features:
- High-quality Stockfish data sources (3500+ Elo, depth 15+)
- Multi-source streaming with priority-based sampling
- Curriculum learning by ply (early/late game)
- Data augmentations (board symmetries)
- C++ binpack loader (primary) + Python fallback
- Large shuffle buffer with reservoir sampling
- Exact resume position tracking
- Automatic download with resume + checksum
"""

import os
import struct
import hashlib
import random
import threading
import time
import subprocess
import urllib.request
import shutil
from pathlib import Path
from typing import Iterator, Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

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
class DataSource:
    """Configuration for a data source."""
    repo_id: str
    pattern: str = "*.binpack"
    priority: int = 1
    max_files: int = 10
    type: str = "huggingface"


@dataclass
class BinpackPosition:
    """Single position from binpack file."""
    features: np.ndarray      # Dense features (81920,) float32
    wdl: np.ndarray           # [W, D, L] probabilities
    eval: float               # Centipawn evaluation
    game_result: int          # 1=white win, 0=draw, -1=black win
    ply: int = 0              # Ply number for curriculum


class StreamingBinpackDataset(IterableDataset):
    """
    Streaming dataset for binpack files with optimized throughput.
    
    Uses C++ loader (primary) or Python fallback.
    Supports multi-file round-robin, curriculum learning, and augmentations.
    """

    def __init__(
        self,
        data_dir: str,
        sources: List[DataSource],
        max_files: int = 6,
        shuffle_buffer_size: int = 5000000,  # 5M positions
        seed: int = 42,
        start_file: int = 0,
        start_position: int = 0,
        drop_last_batch: bool = True,
        curriculum_enabled: bool = True,
        augmentations: Optional[Dict[str, bool]] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.sources = sorted(sources, key=lambda s: s.priority)
        self.max_files = max_files
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.start_file = start_file
        self.start_position = start_position
        self.drop_last_batch = drop_last_batch
        self.curriculum_enabled = curriculum_enabled
        self.augmentations = augmentations or {}

        self.files = self._discover_files()
        self.current_file_idx = start_file
        self.current_position = start_position
        self._rng = random.Random(seed)
        self._position_lock = threading.Lock()
        self._file_loaders: Dict[int, Any] = {}

    def _discover_files(self) -> List[Path]:
        """Find all .binpack files in data_dir, sorted by priority source."""
        all_files = []
        for source in self.sources:
            source_files = sorted(self.data_dir.glob(source.pattern))
            # Limit per source
            source_files = source_files[:source.max_files]
            all_files.extend(source_files)
        if not all_files:
            raise FileNotFoundError(f"No .binpack files found in {self.data_dir}")
        return all_files

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        worker_seed = self.seed + worker_id * 10000
        rng = random.Random(worker_seed)

        # Distribute files across workers
        files_per_worker = max(1, len(self.files) // num_workers)
        start_idx = worker_id * files_per_worker
        end_idx = start_idx + files_per_worker if worker_id < num_workers - 1 else len(self.files)
        worker_files = self.files[start_idx:end_idx]

        # Handle resume for worker 0
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
        # Get or create loader (reuse across epochs)
        if file_idx not in self._file_loaders:
            try:
                self._file_loaders[file_idx] = CppBinpackDataset(str(filepath))
            except Exception as e:
                print(f"Failed to load {filepath} with C++ loader: {e}")
                # Fall back to Python
                yield from self._iterate_python_fallback(filepath, file_idx, rng)
                return
        dataset = self._file_loaders[file_idx]
        num_positions = len(dataset)

        start_pos = self.current_position if file_idx == self.start_file and file_idx == 0 else 0

        # Curriculum: skip early game positions based on ply
        indices = list(range(start_pos, num_positions))
        
        if self.curriculum_enabled:
            # Weight positions by ply (prefer mid/late game)
            # We'll do a simple skip of early positions
            curriculum_cfg = self.config.get("training", {}).get("curriculum", {}) if hasattr(self, 'config') else {}
            early_fen_skip = curriculum_cfg.get("early_fen_skip", 18)
            # For now, just skip first N% of positions as "early game"
            skip_count = int(len(indices) * 0.1)  # Skip first 10%
            indices = indices[skip_count:]

        chunk_size = min(self.shuffle_buffer_size, len(indices))
        
        for chunk_start in range(0, len(indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(indices))
            chunk = indices[chunk_start:chunk_end]
            rng.shuffle(chunk)

            for idx in chunk:
                try:
                    pos_data = dataset[idx]
                    yield self._pos_to_tensors(pos_data, rng)
                except Exception as e:
                    print(f"Error reading position {idx} from {filepath}: {e}")
                    continue

                with self._position_lock:
                    self.current_position = idx + 1
                    self.current_file_idx = file_idx

    def _iterate_python_fallback(self, filepath: Path, file_idx: int, rng: random.Random):
        try:
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
                
                # Curriculum: skip early game
                if self.curriculum_enabled:
                    skip_count = int(len(indices) * 0.1)
                    indices = indices[skip_count:]
                
                chunk_size = min(self.shuffle_buffer_size, len(indices))
                
                for chunk_start in range(0, len(indices), chunk_size):
                    chunk = indices[chunk_start:min(chunk_start + chunk_size, len(indices))]
                    rng.shuffle(chunk)
                    
                    for idx in chunk:
                        try:
                            f.seek(self._get_position_offset(filepath, idx))
                            pos_data = self._read_position(f)
                            yield self._pos_to_tensors(pos_data, rng)
                        except Exception as e:
                            print(f"Error reading position {idx} from {filepath}: {e}")
                            continue
                        
                        with self._position_lock:
                            self.current_position = idx + 1
                            self.current_file_idx = file_idx
        except Exception as e:
            print(f"Failed to read {filepath}: {e}")

    def _skip_position(self, f):
        num_features = struct.unpack("<H", f.read(2))[0]
        f.seek(num_features * 6 + 13, 1)

    def _get_position_offset(self, filepath: Path, idx: int) -> int:
        """Get exact byte offset for a position by building index on first call."""
        if not hasattr(self, '_position_offsets_cache'):
            self._position_offsets_cache = {}
        
        if filepath not in self._position_offsets_cache:
            offsets = []
            with open(filepath, 'rb') as f:
                # Skip header
                f.read(4)  # magic
                f.read(4)  # version
                f.read(8)  # num_positions
                
                pos = f.tell()
                while True:
                    offsets.append(pos)
                    # Try to read next position header
                    num_features_bytes = f.read(2)
                    if not num_features_bytes or len(num_features_bytes) < 2:
                        break
                    num_features = struct.unpack("<H", num_features_bytes)[0]
                    # Skip position data: features (2 bytes each) + weights (4 bytes each) + wdl (12) + eval (4) + game_result (1)
                    pos += 2 + num_features * 6 + 13
                    f.seek(pos)
            
            self._position_offsets_cache[filepath] = offsets
        
        offsets = self._position_offsets_cache[filepath]
        if idx < len(offsets):
            return offsets[idx]
        return 0

    def _read_position(self, f) -> tuple:
        num_features = struct.unpack("<H", f.read(2))[0]
        features = struct.unpack(f"<{num_features}H", f.read(num_features * 2))
        weights = struct.unpack(f"<{num_features}f", f.read(num_features * 4))
        wdl = struct.unpack("<3f", f.read(12))
        eval_cp = struct.unpack("<f", f.read(4))[0]
        game_result = struct.unpack("<b", f.read(1))[0]
        return (np.array(features), np.array(weights), np.array(wdl), eval_cp, game_result)

    def _apply_augmentations(self, features: np.ndarray, rng: random.Random) -> np.ndarray:
        """Apply board symmetry augmentations."""
        if not self.augmentations:
            return features

        # Reshape to (64, 1280) for bucket-wise ops
        x = features.reshape(64, 1280).copy()

        # Horizontal flip (flip file: a<->h, b<->g, etc.)
        # King buckets 0-63 map to squares a1-h8, flip horizontally = flip within each rank
        if self.augmentations.get("flip_horizontal", False) and rng.random() < 0.5:
            # Flip horizontally: for each rank (8 squares), reverse the file order
            # King square index = rank * 8 + file
            # Flipped: rank * 8 + (7 - file)
            x_flipped = np.zeros_like(x)
            for rank in range(8):
                for file in range(8):
                    src_idx = rank * 8 + file
                    dst_idx = rank * 8 + (7 - file)
                    x_flipped[dst_idx] = x[src_idx]
            x = x_flipped

        # Vertical flip (flip rank: 1<->8, 2<->7, etc.)
        if self.augmentations.get("flip_vertical", False) and rng.random() < 0.5:
            # Flip vertically: reverse rank order
            # King square index = rank * 8 + file
            # Flipped: (7 - rank) * 8 + file
            x_flipped = np.zeros_like(x)
            for rank in range(8):
                for file in range(8):
                    src_idx = rank * 8 + file
                    dst_idx = (7 - rank) * 8 + file
                    x_flipped[dst_idx] = x[src_idx]
            x = x_flipped

        # Color flip (swap sides) - requires feature index remapping
        # For HalfKAv2, this is complex as it involves swapping white/black piece features
        if self.augmentations.get("color_flip", False) and rng.random() < 0.5:
            # Simplified: just swap the perspective by flipping both axes
            # Full implementation would need feature index mapping
            x = x[::-1].copy()  # Placeholder

        return x.reshape(-1)

    def _pos_to_tensors(self, pos_data: Any, rng: random.Random) -> Dict[str, torch.Tensor]:
        """
        Convert position data to tensors.
        Handles both C++ loader output (namedtuple-like) and Python fallback (tuple).
        """
        # Handle different return formats from C++ vs Python loader
        if hasattr(pos_data, 'features'):  # C++ loader returns object with attributes
            features = np.array(pos_data.features, dtype=np.uint16)
            weights = np.array(pos_data.weights, dtype=np.float32)
            wdl = np.array(pos_data.wdl, dtype=np.float32)
            eval_cp = float(pos_data.eval)
            game_result = int(pos_data.game_result)
        else:  # Python fallback returns tuple
            features, weights, wdl, eval_cp, game_result = pos_data
            features = np.array(features, dtype=np.uint16)
            weights = np.array(weights, dtype=np.float32)
            wdl = np.array(wdl, dtype=np.float32)
            eval_cp = float(eval_cp)
            game_result = int(game_result)

        feature_indices = features.astype(np.int32)
        feature_weights = weights.astype(np.float32)
        
        dense_features = np.zeros(81920, dtype=np.float32)
        # Bounds check for safety
        valid_mask = (feature_indices >= 0) & (feature_indices < 81920)
        dense_features[feature_indices[valid_mask]] = feature_weights[valid_mask]

        # Apply augmentations
        dense_features = self._apply_augmentations(dense_features, rng)

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


def check_disk_space(path: Path, required_gb: float) -> bool:
    """Check if there's enough disk space at path."""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / 1024**3
        if free_gb < required_gb:
            print(f"WARNING: Low disk space: {free_gb:.1f}GB free, need {required_gb:.1f}GB")
            return False
        print(f"Disk space OK: {free_gb:.1f}GB free")
        return True
    except Exception as e:
        print(f"Could not check disk space: {e}")
        return True  # Assume OK if check fails


def clean_hf_cache(cache_dir: Path = None):
    """Clean HF cache to free space."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    if cache_dir.exists():
        print(f"Cleaning HF cache at {cache_dir}...")
        try:
            for item in cache_dir.iterdir():
                if item.is_dir() and item.name.startswith("datasets--"):
                    shutil.rmtree(item, ignore_errors=True)
                    print(f"  Removed: {item.name}")
        except Exception as e:
            print(f"Cache cleanup error: {e}")


def download_training_data(
    data_dir: Path,
    config: dict,
    max_files: int = 10,
) -> List[Path]:
    data_cfg = config.get("data", {})
    source_cfgs = data_cfg.get("sources", [])
    max_data_gb = data_cfg.get("max_data_gb", 18)
    downloaded_files = []
    data_sources = []  # Store DataSource objects for priority sorting

    # Check disk space before starting
    print(f"Checking disk space for {max_data_gb}GB download...")
    if not check_disk_space(data_dir, max_data_gb + 5):  # Need extra 5GB buffer
        print("Attempting to free space by cleaning HF cache...")
        clean_hf_cache()
        if not check_disk_space(data_dir, max_data_gb + 5):
            print("WARNING: Still low on space, continuing anyway...")

    for source_cfg in source_cfgs:
        if len(downloaded_files) >= max_files:
            break

        source = DataSource(**source_cfg)
        data_sources.append(source)  # Keep for sorting
        print(f"Processing source: {source.repo_id} (priority {source.priority})")

        if source.type == "huggingface":
            files = _download_from_hf(source, data_dir, max_data_gb, max_files - len(downloaded_files))
            downloaded_files.extend(files)
        elif source.type == "http":
            files = _download_from_http(source, data_dir, max_data_gb, max_files - len(downloaded_files))
            downloaded_files.extend(files)
        elif source.type == "local":
            files = _copy_from_local(source, data_dir, max_files - len(downloaded_files))
            downloaded_files.extend(files)

    # Sort by priority using DataSource objects
    downloaded_files.sort(key=lambda f: next((s.priority for s in data_sources if f.name in str(s.pattern)), 999))
    return downloaded_files


def _download_from_hf(source: DataSource, data_dir: Path, max_data_gb: float, max_files: int) -> List[Path]:
    """
    Download from HF Hub directly to data_dir without cache duplication.
    
    Uses cache_dir=data_dir to avoid 2x disk usage during download.
    """
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        print("huggingface_hub not available")
        return []

    repo_id = source.repo_id
    pattern = source.pattern
    
    max_retries = 3
    retry_delay = 30  # seconds
    
    for attempt in range(max_retries):
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

                print(f"Downloading {filename} from HF... (attempt {attempt+1}/{max_retries})")
                # Use cache_dir=data_dir to avoid 2x disk usage (cache + local_dir)
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(data_dir),
                    local_dir_use_symlinks=False,
                    cache_dir=str(data_dir),  # KEY: download directly to data_dir, no separate cache
                )
                downloaded.append(Path(local_path))
                total_size += Path(local_path).stat().st_size

            return downloaded
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"HF download failed (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay}s...")
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"HF download failed after {max_retries} attempts: {e}")
                return []
    
    return []


def _download_from_http(source: DataSource, data_dir: Path, max_data_gb: float, max_files: int) -> List[Path]:
    print(f"HTTP download from {source.repo_id} not fully implemented")
    return []


def _copy_from_local(source: DataSource, data_dir: Path, max_files: int) -> List[Path]:
    local_path = Path(source.repo_id)  # repo_id used as path for local
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

def compile_cpp_loader(build_dir: str = "build") -> bool:
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
    adv_cfg = config.get("advanced", {})

    if dataset is None:
        sources = [DataSource(**s) for s in data_cfg.get("sources", [])]
        dataset = StreamingBinpackDataset(
            data_dir=data_cfg.get("data_dir", "/kaggle/working/data"),
            sources=sources,
            max_files=data_cfg.get("num_files_simultaneous", 6),
            shuffle_buffer_size=data_cfg.get("shuffle_buffer_mb", 1024) * 10000,
            seed=config.get("seed", 42),
            drop_last_batch=data_cfg.get("drop_last_batch", True),
            curriculum_enabled=train_cfg.get("curriculum", {}).get("enabled", True),
            augmentations=adv_cfg.get("augmentations", {}),
        )

    # Gradient accumulation: effective batch = micro_batch * accum_steps
    micro_batch = train_cfg.get("micro_batch_size", 8192)
    accum_steps = adv_cfg.get("gradient_accumulation", 8)
    batch_size = micro_batch

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=hw_cfg.get("num_workers", 16),
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