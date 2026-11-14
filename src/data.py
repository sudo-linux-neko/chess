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

from __future__ import annotations

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
from typing import Iterator, Optional, List, Dict, Any
from dataclasses import dataclass
from collections import deque
from enum import Enum

import torch
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
        streaming_manager: Optional[StreamingDataManager] = None,
        config: Optional[Dict[str, Any]] = None,
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
        self.streaming_manager = streaming_manager
        self.config = config or {}

        self.files = self._discover_files()
        self.current_file_idx = start_file
        self.current_position = start_position
        # Preserve the constructor resume position separately — current_file_idx/
        # current_position get mutated during iteration, so on the next __iter__()
        # call they'd be at end-of-epoch and misinterpreted as a resume position.
        self._resume_start_file = start_file
        self._resume_start_position = start_position
        self._rng = random.Random(seed)
        self._position_lock = threading.Lock()
        self._file_loaders: Dict[int, Any] = {}
        self._max_cached_loaders = 8  # Limit open file handles
        # Track whether resume state has been consumed (prevents epoch 2+
        # from incorrectly using end-of-epoch-1 position as start position)
        self._resume_consumed = False

    def __getstate__(self):
        """Exclude non-picklable threading lock from serialization."""
        state = self.__dict__.copy()
        state.pop('_position_lock', None)
        return state

    def __setstate__(self, state):
        """Restore non-picklable threading lock after deserialization."""
        self.__dict__.update(state)
        self._position_lock = threading.Lock()

    def _discover_files(self) -> List[Path]:
        """Find all .binpack files in data_dir, sorted by priority source."""
        all_files = []
        for source in self.sources:
            source_files = sorted(self.data_dir.glob(source.pattern))
            # Limit per source
            source_files = source_files[:source.max_files]
            all_files.extend(source_files)
        if not all_files:
            # Fall back to globbing all .binpack files if no sources matched
            all_files = sorted(self.data_dir.glob("*.binpack"))
        if not all_files:
            raise FileNotFoundError(f"No .binpack files found in {self.data_dir}")
        return all_files

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        worker_seed = self.seed + worker_id * 10000
        rng = random.Random(worker_seed)

        # DDP sharding across ranks: each GPU trains on unique files.
        # Without this, every rank processes identical data (wastes 50% of data
        # on 2-GPU runs, hurts model quality). Stride sharding is deterministic
        # and requires no communication: rank r gets files[r::world_size].
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank_files = self.files[rank::world_size] if world_size > 1 else list(self.files)

        # Map rank-local index -> global self.files index (for _iterate_* and position tracking)
        def global_index(local_idx: int) -> int:
            return rank + local_idx * world_size if world_size > 1 else local_idx

        # Distribute this rank's files across workers
        files_per_worker = max(1, len(rank_files) // num_workers)
        start_idx = worker_id * files_per_worker
        end_idx = start_idx + files_per_worker if worker_id < num_workers - 1 else len(rank_files)
        worker_files = rank_files[start_idx:end_idx]
        worker_start_idx = start_idx  # index into rank_files

        # Handle resume: use constructor start_file/start_position (preserved
        # separately since current_file_idx/current_position get mutated during
        # iteration and would be at end-of-epoch on the next __iter__ call).
        resume_file = self._resume_start_file
        resume_pos = self._resume_start_position
        _resume_consumed = getattr(self, '_resume_consumed', False)

        if not _resume_consumed and (resume_file > 0 or resume_pos > 0):
            # Does the resume file belong to this rank?
            if world_size <= 1 or (resume_file - rank) % world_size == 0:
                resume_local = (resume_file - rank) // world_size if world_size > 1 else resume_file
                resume_local_pos = resume_pos
            else:
                # Resume file belongs to a different rank (only rank 0 saves checkpoints).
                # Compute this rank's next file in its shard at or after resume_file.
                # Ceiling division: first local index whose global file >= resume_file.
                resume_local = (resume_file - rank + world_size - 1) // world_size if world_size > 1 else resume_file
                resume_local_pos = 0  # Start at beginning of this new file
            if 0 <= resume_local < len(rank_files):
                # Distribute remaining rank-local files from resume_local across workers
                remaining = rank_files[resume_local:]
                rem_per_worker = max(1, len(remaining) // num_workers)
                rem_start = worker_id * rem_per_worker
                rem_end = rem_start + rem_per_worker if worker_id < num_workers - 1 else len(remaining)
                worker_files = remaining[rem_start:rem_end]
                worker_start_idx = resume_local + rem_start
                # Set position for the first file we'll iterate
                self.current_file_idx = global_index(resume_local)
                self.current_position = resume_local_pos
            self._resume_consumed = True
            # Clear consumed resume position so subsequent epochs start fresh
            self._resume_start_file = 0
            self._resume_start_position = 0
        else:
            # Not resuming - reset position for fresh start
            self.current_file_idx = 0
            self.current_position = 0

        for file_idx, filepath in enumerate(worker_files):
            global_file_idx = global_index(worker_start_idx + file_idx)

            if CPP_LOADER_AVAILABLE:
                yield from self._iterate_cpp_loader(filepath, global_file_idx, rng)
            else:
                yield from self._iterate_python_fallback(filepath, global_file_idx, rng)

    def _iterate_cpp_loader(self, filepath: Path, file_idx: int, rng: random.Random):
        # Get or create loader (reuse across epochs) with cache size limit
        if file_idx not in self._file_loaders:
            # Evict oldest if cache full
            if len(self._file_loaders) >= self._max_cached_loaders:
                oldest_idx = next(iter(self._file_loaders))
                del self._file_loaders[oldest_idx]
            try:
                self._file_loaders[file_idx] = CppBinpackDataset(str(filepath))
            except Exception as e:
                print(f"Failed to load {filepath} with C++ loader: {e}")
                # Fall back to Python
                yield from self._iterate_python_fallback(filepath, file_idx, rng)
                return
        dataset = self._file_loaders[file_idx]
        num_positions = len(dataset)

        # Read resume position atomically - current_position/current_file_idx
        # are updated under _position_lock during iteration
        with self._position_lock:
            start_pos = self.current_position if file_idx == self.current_file_idx else 0

        # Curriculum: skip early game positions based on ply
        indices = list(range(start_pos, num_positions))
        
        if self.curriculum_enabled and start_pos == 0:
            # Curriculum: skip early game positions only on fresh start (not resume).
            # early_fen_skipping=18 means skip positions with ply < 18.
            # Without ply info in binpack data, approximate by skipping
            # first (early_fen_skip / max_ply) fraction of positions.
            # On resume (start_pos > 0), we've already passed early game positions,
            # so skipping again would incorrectly skip valid mid/late-game data.
            curriculum_cfg = self.config.get("training", {}).get("curriculum", {})
            early_fen_skip = curriculum_cfg.get("early_fen_skip", 18)
            max_ply = curriculum_cfg.get("max_ply", 64)
            skip_fraction = min(1.0, early_fen_skip / max_ply)
            skip_count = int(len(indices) * skip_fraction)
            indices = indices[skip_count:]

        chunk_size = min(self.shuffle_buffer_size, len(indices))
        
        positions_yielded = 0
        for chunk_start in range(0, len(indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(indices))
            chunk = indices[chunk_start:chunk_end]
            rng.shuffle(chunk)

            for idx in chunk:
                try:
                    pos_data = dataset[idx]
                    yield self._pos_to_tensors(pos_data, rng)
                    positions_yielded += 1
                except Exception as e:
                    print(f"Error reading position {idx} from {filepath}: {e}")
                    # Advance position past failed position so resume doesn't get stuck
                    with self._position_lock:
                        self.current_position = idx + 1
                        self.current_file_idx = file_idx
                    continue

                with self._position_lock:
                    self.current_position = idx + 1
                    self.current_file_idx = file_idx
            
            # Register batch consumption for predictive prefetching
            if self.streaming_manager and positions_yielded > 0:
                self.streaming_manager.register_batch_consumed(filepath, positions_yielded)
                positions_yielded = 0

    def _iterate_python_fallback(self, filepath: Path, file_idx: int, rng: random.Random):
        try:
            with open(filepath, 'rb') as f:
                magic = f.read(4)
                if magic != b"BINP":
                    raise ValueError(f"Invalid binpack magic: {magic}")
                version = struct.unpack("<I", f.read(4))[0]
                num_positions = struct.unpack("<Q", f.read(8))[0]

                # Read resume position atomically - current_position/current_file_idx
                # are updated under _position_lock during iteration
                with self._position_lock:
                    start_pos = self.current_position if file_idx == self.current_file_idx else 0
                
                for _ in range(start_pos):
                    self._skip_position(f)

                indices = list(range(start_pos, num_positions))
                
                # Curriculum: skip early game only on fresh start (not resume).
                # On resume (start_pos > 0), we've already passed early game positions.
                if self.curriculum_enabled and start_pos == 0:
                    curriculum_cfg = self.config.get("training", {}).get("curriculum", {})
                    early_fen_skip = curriculum_cfg.get("early_fen_skip", 18)
                    max_ply = curriculum_cfg.get("max_ply", 64)
                    skip_fraction = min(1.0, early_fen_skip / max_ply)
                    skip_count = int(len(indices) * skip_fraction)
                    indices = indices[skip_count:]
                
                chunk_size = min(self.shuffle_buffer_size, len(indices))
                
                positions_yielded = 0
                for chunk_start in range(0, len(indices), chunk_size):
                    chunk = indices[chunk_start:min(chunk_start + chunk_size, len(indices))]
                    rng.shuffle(chunk)
                    
                    for idx in chunk:
                        try:
                            f.seek(self._get_position_offset(filepath, idx))
                            pos_data = self._read_position(f)
                            yield self._pos_to_tensors(pos_data, rng)
                            positions_yielded += 1
                        except Exception as e:
                            print(f"Error reading position {idx} from {filepath}: {e}")
                            # Advance position past failed position so resume doesn't get stuck
                            with self._position_lock:
                                self.current_position = idx + 1
                                self.current_file_idx = file_idx
                            continue
                        
                        with self._position_lock:
                            self.current_position = idx + 1
                            self.current_file_idx = file_idx
                    
                    # Register batch consumption for predictive prefetching
                    if self.streaming_manager and positions_yielded > 0:
                        self.streaming_manager.register_batch_consumed(filepath, positions_yielded)
                        positions_yielded = 0
        except Exception as e:
            print(f"Failed to read {filepath}: {e}")

    def _skip_position(self, f):
        num_features = struct.unpack("<H", f.read(2))[0]
        # Skip: features(num_features*2) + weights(num_features*4) + wdl(12) + eval(4) + game_result(1)
        f.seek(num_features * 6 + 17, 1)

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
                    pos += 2 + num_features * 6 + 17  # num_features + features + weights + wdl + eval + game_result
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
        """Apply board symmetry augmentations (vectorized NumPy for speed).
        
        NOTE: Augmentations are DISABLED by default for HalfKAv2_hm because
        they require per-bucket feature index remapping (e.g., piece at file f
        must map to file 7-f after horizontal flip), not just bucket reordering.
        Without proper remapping, augmentations produce corrupted training data.
        See train_ultra.yaml advanced.augmentations comments for details.
        """
        if not self.augmentations:
            return features

        # Check if any augmentation is enabled - all are unsafe without proper remapping
        enabled = {k: v for k, v in self.augmentations.items() if v}
        if enabled:
            raise ValueError(
                f"Augmentations are not safely implemented for HalfKAv2_hm: {enabled}. "
                f"Bucket reordering alone corrupts data because features within "
                f"buckets are not remapped. Keep all augmentations disabled "
                f"(set flip_horizontal=false, flip_vertical=false, rotate_180=false, "
                f"color_flip=false in config)."
            )

        return features

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
        
        # Reuse thread-local buffer to avoid per-position allocation (~327KB each)
        input_dim = self.config.get("model", {}).get("input_dim", 81920)
        if not hasattr(self, '_dense_buffer'):
            self._dense_buffer = threading.local()
        if not hasattr(self._dense_buffer, 'buf'):
            self._dense_buffer.buf = np.zeros(input_dim, dtype=np.float32)
        dense_features = self._dense_buffer.buf
        dense_features.fill(0.0)
        
        # Bounds check for safety
        valid_mask = (feature_indices >= 0) & (feature_indices < input_dim)
        dense_features[feature_indices[valid_mask]] = feature_weights[valid_mask]

        # Apply augmentations
        dense_features = self._apply_augmentations(dense_features, rng)

        return {
            "features": torch.from_numpy(dense_features.copy()),  # copy since buffer is reused
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
        # Also set resume start position so __iter__ picks it up on next call
        self._resume_start_file = state.get("current_file_idx", 0)
        self._resume_start_position = state.get("current_position", 0)
        self._resume_consumed = False  # Reset so resume is consumed on next __iter__
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
        
        content_length = int(response.headers.get("Content-Length", 0))
        
        # Check if server honored the range request (206) or ignored it (200)
        if response.status == 206 and existing_size > 0:
            # Server sent remaining bytes only
            total_size = existing_size + content_length
            mode = "ab"
        else:
            # Server sent full file (ignored Range, or no resume requested)
            total_size = content_length
            existing_size = 0  # Reset since we're downloading fresh
            mode = "wb"
            # Clear tmp file since we're overwriting from scratch
            if tmp_dest.exists():
                tmp_dest.unlink()
        
        if max_size_gb and total_size > max_size_gb * 1024**3:
            print(f"File too large: {total_size / 1024**3:.1f}GB > {max_size_gb}GB")
            return False
        
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


class StreamingMode(Enum):
    """Streaming strategy for data management."""
    FIXED_WINDOW = "fixed_window"      # Keep N GB window, rotate files
    ADAPTIVE = "adaptive"              # Adjust based on available space
    ALL_AT_ONCE = "all_at_once"        # Traditional: download all


@dataclass
class StreamingConfig:
    """Configuration for streaming data management."""
    mode: StreamingMode = StreamingMode.ADAPTIVE
    target_gb: float = 5.0             # Target data to keep on disk
    min_gb: float = 2.0                # Minimum before emergency cleanup
    max_gb: float = 8.0                # Maximum before forced rotation
    fallback_gb: float = 3.0           # Fallback if space critical
    keep_recent: int = 2               # Number of recent files to never delete
    auto_cleanup: bool = True          # Auto-delete old files
    cache_dir: Optional[Path] = None   # Separate cache dir


class StreamingDataManager:
    """
    Manages streaming data download/rotation during training.
    
    Features:
    - Keeps only target_gb of data on disk at any time
    - Downloads next files in background while training (predictive prefetching)
    - Auto-cleans old files when space low
    - Falls back to smaller chunks on disk pressure
    - Thread-safe for multi-worker training
    """
    
    def __init__(
        self,
        data_dir: Path,
        config: dict,
        streaming_config: Optional[StreamingConfig] = None,
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.stream_cfg = streaming_config or StreamingConfig()
        self.data_cfg = config.get("data", {})
        self.sources = [DataSource(**s) for s in self.data_cfg.get("sources", [])]
        
        # State
        self._all_files: List[Path] = []          # All available files (downloaded + to-download)
        self._active_files: List[Path] = []       # Currently on disk
        self._downloaded_files: set = set()       # Files ever downloaded
        self._actual_paths: Dict[Path, Path] = {}  # symlink -> actual file path mapping
        self._lock = threading.RLock()
        self._download_queue: deque = deque()
        self._download_thread: Optional[threading.Thread] = None
        self._stop_download = threading.Event()
        self._download_errors: List[str] = []
        
        # Predictive prefetching state
        self._file_consumption_rates: Dict[Path, float] = {}  # positions/sec per file
        self._last_consumption_update = time.time()
        self._total_positions_consumed = 0
        self._prefetch_threshold_pct = 0.5  # Start next download at 50% consumption
        self._min_prefetch_gb = 2.0  # Minimum GB to keep as buffer
        
        # Discover all potential files from HF (without downloading)
        self._discover_all_files()
        
        # Initial download to fill target_gb
        self._ensure_initial_data()
        
        # Start background downloader
        if self.stream_cfg.auto_cleanup:
            self._start_background_downloader()

    def __getstate__(self):
        """Exclude non-picklable threading objects from serialization."""
        state = self.__dict__.copy()
        state.pop('_lock', None)
        state.pop('_stop_download', None)
        state.pop('_download_thread', None)
        return state

    def __setstate__(self, state):
        """Restore non-picklable threading objects after deserialization."""
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._stop_download = threading.Event()
        self._download_thread = None

    def _discover_all_files(self):
        """List all available files from HF sources without downloading."""
        print("Discovering available training files...")
        for source in self.sources:
            if source.type != "huggingface":
                continue
            try:
                from huggingface_hub import list_repo_files
                files = list_repo_files(repo_id=source.repo_id, repo_type="dataset")
                binpack_files = [f for f in files if f.endswith(".binpack")]
                binpack_files.sort()
                
                # Limit per source
                binpack_files = binpack_files[:source.max_files]
                
                for filename in binpack_files:
                    # Always use data_dir/filename as the canonical tracking path
                    # (the symlink path that create_dataloader expects).
                    # actual_path is where hf_hub_download actually stores the file.
                    link_path = self.data_dir / filename
                    actual_path = self.data_dir / source.repo_id / filename
                    
                    # If symlink already exists AND target is valid - mark as downloaded/active
                    # (exists() returns False for broken symlinks, so they'll be re-downloaded)
                    if link_path.exists():
                        self._all_files.append(link_path)
                        self._downloaded_files.add(link_path)
                        self._active_files.append(link_path)
                        self._actual_paths[str(link_path)] = actual_path if actual_path.exists() else link_path
                    # If actual file exists but no symlink yet - will be fixed on
                    # first _download_file call. Track as all_files but not yet downloaded.
                    elif actual_path.exists():
                        self._all_files.append(link_path)
                    # File not yet downloaded
                    else:
                        self._all_files.append(link_path)
                
                print(f"  {source.repo_id}: {len(binpack_files)} files available")
            except Exception as e:
                print(f"  Failed to list {source.repo_id}: {e}")
        
        print(f"Total available files: {len(self._all_files)}")
        print(f"Already downloaded: {len(self._downloaded_files)}")
    
    def _get_active_size_gb(self) -> float:
        """Get total size of active files in GB."""
        with self._lock:
            total = sum(f.stat().st_size for f in self._active_files if f.exists())
            return total / 1024**3
    
    def _ensure_initial_data(self):
        """Download just enough to start training immediately.
        
        Downloads 1 priority file (enough to begin), then lets the
        background downloader prefetch the rest. Training never waits
        for full target_gb.
        """
        target = self.stream_cfg.target_gb
        current = self._get_active_size_gb()
        
        if current >= target:
            print(f"Already have {current:.1f}GB active data (target: {target}GB)")
            return
        
        if not self._all_files:
            print("No files available to download")
            return
        
        # Sort by priority, download just 1 file to start
        import fnmatch
        priority_files = sorted(
            [f for f in self._all_files if f not in self._downloaded_files],
            key=lambda f: next((s.priority for s in self.sources if fnmatch.fnmatch(f.name, s.pattern)), 999)
        )
        
        if priority_files:
            print(f"Downloading first file to start training (target: {target}GB, background will prefetch rest)...")
            try:
                self._download_file(priority_files[0])
            except Exception as e:
                print(f"Initial download failed: {e}")
            
            current = self._get_active_size_gb()
            if current > 0:
                print(f"Ready to train with {current:.1f}GB on disk. Background downloading rest...")
            else:
                print("WARNING: No data downloaded yet, will retry in background")
        else:
            print("All files already downloaded")
    
    def _download_file(self, filepath: Path) -> bool:
        """Download a single file with space management."""
        with self._lock:
            if filepath in self._downloaded_files:
                # File was downloaded before - verify it still exists (cleanup may have deleted it)
                if filepath.exists():
                    return True
                # File was deleted by cleanup - remove from tracking and re-download
                self._downloaded_files.discard(filepath)
                try:
                    self._active_files.remove(filepath)
                except ValueError:
                    pass
        
        # Check space before download
        if not self._ensure_space_for_download(filepath):
            return False
        
        # Find source for this file
        source = None
        import fnmatch
        for s in self.sources:
            if fnmatch.fnmatch(filepath.name, s.pattern) or filepath.match(s.pattern):
                source = s
                break
        
        if not source:
            print(f"No source found for {filepath.name}")
            return False
        
        # Pre-check file size against available disk space
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.dataset_info(repo_id=source.repo_id, files_metadata=True)
            for sibling in info.siblings:
                if sibling.rfilename == filepath.name and sibling.size:
                    stat = shutil.disk_usage(self.data_dir)
                    free_gb = stat.free / 1024**3
                    needed_gb = sibling.size / 1024**3 + 1.0  # 1GB safety margin
                    if free_gb < needed_gb:
                        print(f"  Skipping {filepath.name} (~{needed_gb:.1f}GB needed, {free_gb:.1f}GB free)")
                        return False
                    break
        except Exception:
            pass  # If we can't get size, proceed and check as we go
        
        print(f"Downloading {filepath.name} ({source.repo_id})...")
        
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id=source.repo_id,
                filename=filepath.name,
                repo_type="dataset",
                local_dir=str(self.data_dir),
                force_download=False,
            )
            
            actual_path = Path(local_path)
            
            # Create symlink at data_dir/filename for consistent path tracking
            # hf_hub_download with local_dir stores at data_dir/<repo_id>/<filename>
            # but _discover_all_files uses data_dir/<filename>, so we symlink
            link_path = _create_data_symlink(self.data_dir, filepath.name, actual_path)
            
            with self._lock:
                self._downloaded_files.add(link_path)
                self._active_files.append(link_path)
                # Track actual path for cleanup
                self._actual_paths[str(link_path)] = actual_path
            
            size_gb = actual_path.stat().st_size / 1024**3
            print(f"Downloaded: {filepath.name} ({size_gb:.2f}GB)")
            return True
            
        except Exception as e:
            print(f"Failed to download {filepath.name}: {e}")
            with self._lock:
                self._download_errors.append(f"{filepath.name}: {e}")
            return False
    
    def _ensure_space_for_download(self, filepath: Path) -> bool:
        """Ensure enough space for download, cleanup if needed."""
        required = 5.0  # Estimate 5GB per file
        
        # Check current free space
        try:
            stat = shutil.disk_usage(self.data_dir)
            free_gb = stat.free / 1024**3
            
            if free_gb > required + 2:  # 2GB buffer
                return True
            
            print(f"Low space: {free_gb:.1f}GB free, need {required}GB for {filepath.name}")
            
            # Try cleanup
            self._cleanup_old_files(required + 2)
            stat = shutil.disk_usage(self.data_dir)
            free_gb = stat.free / 1024**3
            if free_gb > required:
                return True
            
            # Fallback: try with smaller target
            print(f"Falling back to {self.stream_cfg.fallback_gb}GB target...")
            self.stream_cfg.target_gb = self.stream_cfg.fallback_gb
            self._cleanup_old_files(self.stream_cfg.fallback_gb + 2)
            
            stat = shutil.disk_usage(self.data_dir)
            free_gb = stat.free / 1024**3
            return free_gb > required
            
        except Exception as e:
            print(f"Space check failed: {e}")
            return False  # Don't assume OK - skip download to avoid disk-full
    
    def _cleanup_old_files(self, target_free_gb: float) -> bool:
        """Delete oldest files to free space, keeping recent ones."""
        with self._lock:
            # Sort by modification time (oldest first)
            deletable = sorted(
                [f for f in self._active_files if f.exists()],
                key=lambda f: f.stat().st_mtime
            )
            
            # Keep recent files
            keep_count = self.stream_cfg.keep_recent
            deletable = deletable[:-keep_count] if len(deletable) > keep_count else []
            
            if not deletable:
                print("No files to delete (keeping recent)")
                return False
            
            freed = 0
            for f in deletable:
                try:
                    size = f.stat().st_size
                    # Remove symlink
                    f.unlink(missing_ok=True)
                    # Also remove actual file if tracked
                    actual = self._actual_paths.get(str(f))
                    if actual is not None:
                        try:
                            actual.unlink(missing_ok=True)
                        except Exception:
                            pass
                        del self._actual_paths[str(f)]
                    self._active_files.remove(f)
                    self._downloaded_files.discard(f)  # Allow re-download after cleanup
                    freed += size
                    print(f"  Deleted: {f.name} ({size/1024**3:.2f}GB)")
                    
                    if freed / 1024**3 >= target_free_gb:
                        break
                except Exception as e:
                    print(f"  Failed to delete {f.name}: {e}")
            
            print(f"Freed {freed/1024**3:.2f}GB")
            return freed > 0
    
    def _start_background_downloader(self):
        """Start background thread to download next files."""
        self._download_thread = threading.Thread(
            target=self._background_downloader,
            daemon=True,
            name="DataDownloader"
        )
        self._download_thread.start()
    
    def _background_downloader(self):
        """
        Background thread: PREDICTIVELY download next files.
        
        Strategy:
        - Monitor consumption rate of active files
        - Start next download when current file ~50% consumed
        - Always keep min_prefetch_gb buffer ready
        - Download in priority order
        """
        print("Background downloader started (predictive prefetching)")
        
        while not self._stop_download.is_set():
            try:
                current_gb = self._get_active_size_gb()
                target = self.stream_cfg.target_gb
                
                # Update consumption rates
                self._update_consumption_rates()
                
                # Check if we need to prefetch next file
                should_prefetch = self._should_prefetch_next()
                
                # If under target OR should prefetch, download next
                if current_gb < target or should_prefetch:
                    with self._lock:
                        # Find next undownloaded file by priority
                        next_file = None
                        for f in self._all_files:
                            if f not in self._downloaded_files:
                                next_file = f
                                break
                    
                    if next_file:
                        print(f"Background: prefetching next file ({current_gb:.1f}/{target}GB, rate={self._get_avg_consumption_rate():.1f} pos/s)")
                        self._download_file(next_file)
                    else:
                        print("Background: all files downloaded")
                        break
                
                # Check for space pressure
                elif current_gb > self.stream_cfg.max_gb:
                    print(f"Background: over max ({current_gb:.1f} > {self.stream_cfg.max_gb}GB), cleaning...")
                    self._cleanup_old_files(current_gb - target)
                
                # Dynamic sleep: more frequent when close to needing prefetch
                sleep_time = self._calculate_sleep_time()
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Background downloader error: {e}")
                time.sleep(60)
    
    def _update_consumption_rates(self):
        """Estimate how fast training consumes each file."""
        now = time.time()
        dt = now - self._last_consumption_update
        if dt < 10:  # Update at most every 10 seconds
            return
        
        with self._lock:
            for f in self._active_files:
                if f.exists():
                    # Track file access time for LRU
                    try:
                        mtime = f.stat().st_mtime
                        # If file was accessed recently, it's being consumed
                        if now - mtime < 60:  # Accessed in last minute
                            if f not in self._file_consumption_rates:
                                self._file_consumption_rates[f] = 0
                            # Rough estimate: if accessed, consumption > 0
                            self._file_consumption_rates[f] = max(
                                self._file_consumption_rates[f], 1000
                            )
                    except Exception:
                        pass
        
        self._last_consumption_update = now
    
    def _get_avg_consumption_rate(self) -> float:
        """Get average consumption rate across active files."""
        with self._lock:
            if not self._file_consumption_rates:
                return 50000  # Default estimate: 50k pos/s
            return sum(self._file_consumption_rates.values()) / len(self._file_consumption_rates)
    
    def _should_prefetch_next(self) -> bool:
        """
        Determine if we should start downloading the next file.
        
        Returns True if:
        - Current active files are being consumed (consumption rate > 0)
        - We have less than min_prefetch_gb buffer
        - Or current file is > prefetch_threshold_pct consumed
        
        Never returns True if we're already at or over max_gb.
        """
        with self._lock:
            if not self._active_files:
                return True  # No files at all
            
            current_gb = self._get_active_size_gb()
            
            # Never prefetch if already at or over max_gb
            if current_gb >= self.stream_cfg.max_gb:
                return False
            
            # Check if we have enough buffer
            if current_gb < self._min_prefetch_gb:
                return True
            
            # Check consumption rate - if data is being consumed, prefetch
            avg_rate = self._get_avg_consumption_rate()
            if avg_rate > 1000:  # Data is being consumed
                # Estimate time until we need next file
                # Rough: at 50k pos/s, 5GB (~50M pos) lasts ~1000s
                # Start download 500s before = 50% threshold
                return True
        
        return False
    
    def _calculate_sleep_time(self) -> int:
        """Dynamic sleep based on urgency."""
        current_gb = self._get_active_size_gb()
        target = self.stream_cfg.target_gb
        
        if current_gb < self._min_prefetch_gb:
            return 10  # Urgent: check every 10s
        elif current_gb < target * 0.7:
            return 20  # Getting low: check every 20s
        elif self._should_prefetch_next():
            return 15  # Prefetch soon: check every 15s
        else:
            return 30  # Normal: check every 30s
    
    def register_batch_consumed(self, filepath: Path, num_positions: int):
        """Register that a batch was consumed from a file for rate tracking."""
        with self._lock:
            self._total_positions_consumed += num_positions
            if filepath not in self._file_consumption_rates:
                self._file_consumption_rates[filepath] = 0
            # Update running average
            self._file_consumption_rates[filepath] = (
                self._file_consumption_rates[filepath] * 0.9 + num_positions * 0.1
            )
    
    def get_active_files(self) -> List[Path]:
        """Get currently available files for training."""
        with self._lock:
            return [f for f in self._active_files if f.exists()]
    
    def shutdown(self):
        """Stop background downloader."""
        self._stop_download.set()
        if self._download_thread:
            self._download_thread.join(timeout=10)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming stats."""
        with self._lock:
            return {
                "active_files": len(self._active_files),
                "active_gb": self._get_active_size_gb(),
                "total_available": len(self._all_files),
                "downloaded": len(self._downloaded_files),
                "target_gb": self.stream_cfg.target_gb,
                "avg_consumption_rate": self._get_avg_consumption_rate(),
                "total_positions_consumed": self._total_positions_consumed,
                "download_errors": len(self._download_errors),
            }


def create_streaming_manager(
    data_dir: Path,
    config: dict,
    streaming_config: Optional[StreamingConfig] = None,
) -> StreamingDataManager:
    """Factory function for streaming data manager."""
    return StreamingDataManager(data_dir, config, streaming_config)


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
    print(f"Checking disk space for up to {max_data_gb}GB download...")
    free_gb = check_disk_space(data_dir, max_data_gb + 2)
    if not free_gb:
        print("Attempting to free space by cleaning HF cache...")
        clean_hf_cache()
        free_gb = check_disk_space(data_dir, max_data_gb + 2)
        if not free_gb:
            print("WARNING: Low disk space - will download minimal files only")
            max_data_gb = min(max_data_gb, 5.0)  # Cap at 5GB if space is tight

    for source_cfg in source_cfgs:
        if len(downloaded_files) >= max_files:
            break

        source = DataSource(**source_cfg)
        data_sources.append(source)  # Keep for sorting
        print(f"Processing source: {source.repo_id} (priority {source.priority})")

        if source.type == "huggingface":
            files = _download_from_hf(source, data_dir, max_data_gb, max_files - len(downloaded_files))
            downloaded_files.extend(files)
        elif source.type == "local":
            files = _copy_from_local(source, data_dir, max_files - len(downloaded_files))
            downloaded_files.extend(files)

        # Re-check disk after each source
        free_gb = check_disk_space(data_dir, 2)
        if not free_gb:
            print("Disk space critically low, stopping download")
            break

    # Sort by priority using DataSource objects
    import fnmatch
    downloaded_files.sort(key=lambda f: next((s.priority for s in data_sources if fnmatch.fnmatch(f.name, s.pattern)), 999))
    return downloaded_files


def _create_data_symlink(data_dir: Path, filename: str, actual_path: Path) -> Path:
    """Create symlink at data_dir/filename pointing to actual_path.
    
    hf_hub_download with local_dir stores at data_dir/<repo_id>/<filename>
    but callers use data_dir.glob("*.binpack") which expects data_dir/<filename>.
    Falls back to actual_path if symlinks are not supported.
    """
    link_path = data_dir / filename
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink(missing_ok=True)
        link_path.symlink_to(actual_path)
    except Exception:
        # Symlinks not supported, use actual path
        link_path = actual_path
    return link_path


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
            
            # Get file sizes from HF API to pre-check disk space
            file_sizes = {}
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                info = api.dataset_info(repo_id=repo_id, files_metadata=True)
                for sibling in info.siblings:
                    if sibling.rfilename in binpack_files:
                        file_sizes[sibling.rfilename] = sibling.size or 0
            except Exception:
                pass  # If we can't get sizes, download and check as we go
            
            for filename in binpack_files[:max_files]:
                dest = data_dir / filename
                if dest.exists():
                    downloaded.append(dest)
                    total_size += dest.stat().st_size
                    print(f"  Already have: {filename}")
                    continue

                # Pre-check: skip files that won't fit on disk
                estimated_size = file_sizes.get(filename, 0)
                if estimated_size == 0:
                    # No size info - estimate based on max_data_gb / max_files
                    estimated_size = (max_data_gb * 1024**3) / max_files
                
                stat = shutil.disk_usage(data_dir)
                free_gb = stat.free / 1024**3
                needed_gb = estimated_size / 1024**3 + 1.0  # 1GB safety margin
                
                if free_gb < needed_gb:
                    print(f"  Skipping {filename} (~{needed_gb:.1f}GB needed, {free_gb:.1f}GB free)")
                    continue
                
                if free_gb < 1.0:
                    print(f"  Disk space low ({free_gb:.1f}GB free), stopping download")
                    break

                if total_size > max_data_gb * 1024**3:
                    print(f"  Reached {max_data_gb}GB limit, stopping download")
                    break

                print(f"  Downloading {filename} from HF... (attempt {attempt+1}/{max_retries})")
                # Use local_dir only (no cache_dir to avoid conflicts and 2x disk usage)
                try:
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type="dataset",
                        local_dir=str(data_dir),
                        force_download=False,
                    )
                    actual_path = Path(local_path)
                    
                    # Create symlink at data_dir/filename for consistent path tracking
                    # hf_hub_download with local_dir stores at data_dir/<repo_id>/<filename>
                    # but callers use data_dir.glob("*.binpack") which expects data_dir/<filename>
                    link_path = _create_data_symlink(data_dir, filename, actual_path)
                    
                    downloaded.append(link_path)
                    file_size = actual_path.stat().st_size
                    total_size += file_size
                    print(f"  Downloaded: {filename} ({file_size / 1024**3:.2f}GB)")
                except Exception as e:
                    # Skip missing/404 files gracefully - try next file
                    print(f"  Skipping {filename} (not found: {e})")
                    continue

            return downloaded
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"HF download failed (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"HF download failed after {max_retries} attempts: {e}")
                return []
    
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
    
    # Ensure __init__.py exists for import (in case dir pre-existed)
    init_py = cpp_loader_dir / "__init__.py"
    if not init_py.exists():
        init_py.write_text('try:\n    from .cpp_loader import BinpackDataset\nexcept ImportError:\n    pass\n')

    build_path = cpp_loader_dir / build_dir
    build_path.mkdir(parents=True, exist_ok=True)

    # Set CMAKE_PREFIX_PATH so CMake can find pybind11
    env = os.environ.copy()
    try:
        import pybind11
        cmake_dir = pybind11.get_cmake_dir()
        env["CMAKE_PREFIX_PATH"] = cmake_dir + os.pathsep + env.get("CMAKE_PREFIX_PATH", "")
    except ImportError:
        print("  pybind11 not installed - C++ loader will not compile")
        return False

    try:
        result = subprocess.run(
            ["cmake", ".."],
            cwd=build_path,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
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
            env=env,
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

    # Create __init__.py so the compiled .so can be imported as `from cpp_loader import BinpackDataset`
    init_content = '''
try:
    from .cpp_loader import BinpackDataset
except ImportError:
    pass
'''
    (cpp_loader_dir / "__init__.py").write_text(init_content)

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
            pos = file.tellg(); pos += num_features * 6 + 17; file.seekg(pos);
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
Position BinpackDataset::operator[](size_t idx) const { return pimpl_->read_position(idx); }
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
                py::array_t<uint16_t>(pos.features),
                py::array_t<float>(pos.weights),
                py::array_t<float>(std::vector<float>(pos.wdl.begin(), pos.wdl.end())),
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
    streaming_manager: Optional[StreamingDataManager] = None,
    is_train: bool = True,
    start_file: int = 0,
    start_position: int = 0,
) -> DataLoader:
    hw_cfg = config.get("hardware", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    adv_cfg = config.get("advanced", {})
    
    # Get streaming config from data config
    stream_cfg_dict = data_cfg.get("streaming", {})
    use_streaming = stream_cfg_dict.get("enabled", False)

    if dataset is None:
        sources = [DataSource(**s) for s in data_cfg.get("sources", [])]
        
        if use_streaming and streaming_manager is not None:
            # Use streaming manager's active files - wait for background downloader
            active_files = streaming_manager.get_active_files()
            wait_seconds = 0
            max_wait = 120  # Wait up to 2 minutes for first file
            while not active_files and wait_seconds < max_wait:
                print(f"  Waiting for background downloader... ({wait_seconds}s/{max_wait}s)")
                time.sleep(10)
                wait_seconds += 10
                active_files = streaming_manager.get_active_files()
            
            if not active_files:
                # Fall back: try non-streaming mode with existing files
                print("  WARNING: No files from streaming manager, trying non-streaming fallback...")
                binpack_files = list(Path(data_cfg.get("data_dir", "/kaggle/working/data")).glob("*.binpack"))
                if binpack_files:
                    print(f"  Found {len(binpack_files)} existing files, using non-streaming mode")
                    dataset = StreamingBinpackDataset(
                        data_dir=data_cfg.get("data_dir", "/kaggle/working/data"),
                        sources=sources,
                        max_files=min(len(binpack_files), data_cfg.get("num_files_simultaneous", 6)),
                        shuffle_buffer_size=data_cfg.get("shuffle_buffer_mb", 1024) * 10000,
                        seed=config.get("seed", 42),
                        start_file=start_file,
                        start_position=start_position,
                        drop_last_batch=data_cfg.get("drop_last_batch", True),
                        curriculum_enabled=train_cfg.get("curriculum", {}).get("enabled", True),
                        augmentations=adv_cfg.get("augmentations", {}),
                        config=config,
                    )
                else:
                    raise RuntimeError("No .binpack files found anywhere - cannot start training")
            else:
                # Create a custom dataset that uses the streaming manager's files
                dataset = StreamingBinpackDataset(
                    data_dir=data_cfg.get("data_dir", "/kaggle/working/data"),
                    sources=sources,
                    max_files=len(active_files),
                    shuffle_buffer_size=data_cfg.get("shuffle_buffer_mb", 1024) * 10000,
                    seed=config.get("seed", 42),
                    start_file=start_file,
                    start_position=start_position,
                    drop_last_batch=data_cfg.get("drop_last_batch", True),
                    curriculum_enabled=train_cfg.get("curriculum", {}).get("enabled", True),
                    augmentations=adv_cfg.get("augmentations", {}),
                    streaming_manager=streaming_manager,  # Pass for predictive prefetching
                    config=config,  # Pass config for curriculum settings
                )
                # Override file discovery to use streaming manager's active files
                # active_files is a subset (currently downloaded files). _update_data_position()
                # computes indices relative to dataset.files (which is active_files in streaming
                # mode), so start_file/start_position from training state ARE valid indices into
                # active_files. Keep them from the constructor instead of resetting to 0,
                # otherwise resume position is lost and training restarts from the beginning.
                dataset.files = active_files
                # Clamp resume position to active file bounds (files may have been
                # cleaned up between checkpoint and resume)
                if dataset.current_file_idx >= len(active_files):
                    dataset.current_file_idx = max(0, len(active_files) - 1)
                    dataset.current_position = 0
                    # Also clear resume start position so __iter__ doesn't keep
                    # trying to resume from an out-of-range file every epoch
                    dataset._resume_start_file = 0
                    dataset._resume_start_position = 0
                    dataset._resume_consumed = True
                print(f"Streaming mode: using {len(active_files)} active files ({streaming_manager._get_active_size_gb():.1f}GB)")
        else:
            # Non-streaming: check for existing data files first
            binpack_files = list(Path(data_cfg.get("data_dir", "/kaggle/working/data")).glob("*.binpack"))
            if not binpack_files:
                print("  ERROR: No .binpack files found - cannot start training in non-streaming mode")
                print("  Please download data first with --download-data or enable streaming")
                raise RuntimeError("No .binpack files found for non-streaming training")
            print(f"  Found {len(binpack_files)} data files")
            dataset = StreamingBinpackDataset(
                data_dir=data_cfg.get("data_dir", "/kaggle/working/data"),
                sources=sources,
                max_files=data_cfg.get("num_files_simultaneous", 6),
                shuffle_buffer_size=data_cfg.get("shuffle_buffer_mb", 1024) * 10000,
                seed=config.get("seed", 42),
                start_file=start_file,
                start_position=start_position,
                drop_last_batch=data_cfg.get("drop_last_batch", True),
                curriculum_enabled=train_cfg.get("curriculum", {}).get("enabled", True),
                augmentations=adv_cfg.get("augmentations", {}),
                config=config,  # Pass config for curriculum settings
            )

    # Gradient accumulation handled by PyTorch Lightning via accumulate_grad_batches.
    # DataLoader batch_size is the per-step (micro) batch size only.
    micro_batch = train_cfg.get("micro_batch_size", 8192)
    batch_size = micro_batch

    # StreamingBinpackDataset is an IterableDataset that handles its own
    # DDP + worker sharding in __iter__(). Forcing num_workers=0 avoids:
    #   1. Double-sharding (DataLoader workers + dataset's internal sharding)
    #   2. Resume duplication: each pickled worker copy starts with
    #      _resume_consumed=False, so all workers resume from the same position
    #   3. Unnecessary process overhead for a disk-reading dataset
    num_workers = 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory=hw_cfg.get("pin_memory", True),
        drop_last=is_train and data_cfg.get("drop_last_batch", True),
        worker_init_fn=None,
        multiprocessing_context=None,
    )
