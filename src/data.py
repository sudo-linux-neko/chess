# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Streaming Stockfish binpack data.

All decoding and HalfKAv2_hm feature extraction happens in Rust
(``nnue_core``); this module only downloads shards and converts the produced
numpy batches to PyTorch tensors. No positions are ever materialised in
Python, and the stream is exactly resumable at (file, chunk, entry).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

import nnue_core


def default_cache_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/data")
    return Path(os.environ.get("NNUE_DATA_DIR", "./data"))


@dataclass
class DataConfig:
    repo: str = "vondele/master-binpacks_relabel"
    repo_type: str = "dataset"
    files: list[str] = field(default_factory=lambda: ["fishpack32.relabel-BT4-tf13tune.binpack"])
    cache_dir: str = ""
    cycle: bool = True

    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir) if self.cache_dir else default_cache_dir()


def ensure_files(cfg: DataConfig, verbose: bool = True) -> list[str]:
    """Download missing binpacks and return local paths."""
    cache = cfg.resolved_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name in cfg.files:
        local = cache / Path(name).name
        if not local.exists() or local.stat().st_size == 0:
            from huggingface_hub import hf_hub_download

            if verbose:
                print(f"[data] downloading {cfg.repo}/{name} -> {local}")
            hf_hub_download(
                repo_id=cfg.repo,
                filename=name,
                repo_type=cfg.repo_type,
                local_dir=str(cache),
            )
        if not local.exists():
            raise FileNotFoundError(f"binpack not found after download: {local}")
        paths.append(str(local))
    return paths


def shard_start(path: str, world_size: int, rank: int) -> int:
    """Chunk-aligned byte offset for this rank's contiguous shard."""
    size = nnue_core.file_size(path)
    if world_size <= 1:
        return 0
    approx = size * rank // world_size
    return nnue_core.find_chunk_offset(path, approx)


@dataclass
class StreamState:
    file_index: int = 0
    byte_offset: int = 0
    skip_entries: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "file_index": self.file_index,
            "byte_offset": self.byte_offset,
            "skip_entries": self.skip_entries,
        }

    @classmethod
    def from_any(cls, value: Any) -> "StreamState":
        if value is None:
            return cls()
        if isinstance(value, StreamState):
            return value
        return cls(
            int(value.get("file_index", 0)),
            int(value.get("byte_offset", 0)),
            int(value.get("skip_entries", 0)),
        )


class BinpackStream:
    """Endless (optionally cycling) stream of binpack batches for one rank."""

    def __init__(
        self,
        paths: list[str],
        world_size: int = 1,
        rank: int = 0,
        state: StreamState | None = None,
        cycle: bool = True,
    ):
        if not paths:
            raise ValueError("no binpack paths")
        self.paths = paths
        self.world_size = max(1, world_size)
        self.rank = rank
        self.cycle = cycle
        self.state = state or StreamState()
        self._reader = None
        self._open()

    def _open(self) -> None:
        path = self.paths[self.state.file_index]
        if self.state.byte_offset == 0:
            self.state.byte_offset = shard_start(path, self.world_size, self.rank)
        self._reader = nnue_core.BinpackReader(
            path, self.state.byte_offset, self.state.skip_entries
        )

    def _advance_file(self) -> bool:
        if self.state.file_index + 1 < len(self.paths):
            self.state.file_index += 1
            self.state.byte_offset = shard_start(
                self.paths[self.state.file_index], self.world_size, self.rank
            )
            self.state.skip_entries = 0
            self._open()
            return True
        if self.cycle:
            self.state.file_index = 0
            self.state.byte_offset = shard_start(
                self.paths[0], self.world_size, self.rank
            )
            self.state.skip_entries = 0
            self._open()
            return True
        return False

    def next_cpu(self, n: int) -> dict[str, torch.Tensor]:
        while True:
            raw = self._reader.next_batch(n)
            if raw["count"] > 0:
                break
            if not self._advance_file():
                raise StopIteration
        byte, skip = self._reader.cursor()
        self.state.byte_offset = byte
        self.state.skip_entries = skip

        features = torch.from_numpy(np.asarray(raw["features"])).to(torch.int32)
        stm = torch.from_numpy(np.asarray(raw["stm"])).to(torch.int64)
        bucket = torch.from_numpy(np.asarray(raw["bucket"])).to(torch.int64)
        score = torch.from_numpy(np.asarray(raw["score"])).to(torch.float32)
        result = torch.from_numpy(np.asarray(raw["result"])).to(torch.float32)
        return {
            "features": features,
            "stm": stm,
            "bucket": bucket,
            "score": score,
            "result": result,
        }

    def state_dict(self) -> dict[str, int]:
        # The current reader cursor is authoritative.
        if self._reader is not None:
            byte, skip = self._reader.cursor()
            self.state.byte_offset = byte
            self.state.skip_entries = skip
        return self.state.to_dict()


def move_batch_to(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    # Pinning per batch would allocate new page-locked buffers every step;
    # plain async copies are faster for these small batches.
    nb = device.type == "cuda"
    return {k: v.to(device, non_blocking=nb) for k, v in batch.items()}
