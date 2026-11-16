# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Atomic checkpoints + optional background Hugging Face upload.

Resume is exact: model, optimizer, step counters, LR schedule, RNG state and
each rank's binpack (file, chunk, entry) cursor are stored together.
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def atomic_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    data_states: list[dict[str, int]],
    config: dict[str, Any],
    best_loss: float = float("inf"),
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format": 2,
        "author": "Vaibhav",
        "license": "GPL-3.0-or-later",
        "created": int(time.time()),
        "model": _strip_compile_prefix(
            (model.module if hasattr(model, "module") else model).state_dict()
        ),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "total_steps": total_steps,
        "data_states": data_states,
        "config": config,
        "best_loss": best_loss,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if extra:
        payload.update(extra)
    atomic_save(payload, path)


def _strip_compile_prefix(sd: dict[str, Any]) -> dict[str, Any]:
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def load_checkpoint(path: str | Path, map_location="cpu") -> dict[str, Any]:
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(obj, dict) or obj.get("format") != 2:
        raise ValueError(f"not a v2 trainer checkpoint: {path}")
    obj["model"] = _strip_compile_prefix(obj["model"])
    return obj


def try_download_checkpoint(
    local_path: str | Path,
    repo_id: str,
    filename: str,
    repo_type: str = "model",
) -> str | None:
    """Download a checkpoint from HF if it is not already present locally."""
    local = Path(local_path)
    if local.exists():
        return str(local)
    try:
        from huggingface_hub import hf_hub_download

        got = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
        local.parent.mkdir(parents=True, exist_ok=True)
        if Path(got).resolve() != local.resolve():
            shutil.copy2(got, local)
        return str(local)
    except Exception as e:  # noqa: BLE001
        print(f"[hf] no checkpoint to resume ({e})")
        return None


@dataclass
class _UploadJob:
    local: str
    repo_id: str
    path_in_repo: str
    repo_type: str


class HFUploader:
    """Single background thread; never blocks the training step."""

    def __init__(self) -> None:
        self.q: queue.Queue[_UploadJob | None] = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._run, daemon=True, name="hf-upload")
        self.thread.start()

    def submit(self, local: str, repo_id: str, path_in_repo: str, repo_type: str = "model") -> None:
        job = _UploadJob(local, repo_id, path_in_repo, repo_type)
        try:
            self.q.put_nowait(job)
        except queue.Full:
            print("[hf] upload queue full, skipping this checkpoint upload")

    def _run(self) -> None:
        from huggingface_hub import HfApi

        api = HfApi()
        while True:
            job = self.q.get()
            if job is None:
                return
            try:
                api.upload_file(
                    path_or_fileobj=job.local,
                    path_in_repo=job.path_in_repo,
                    repo_id=job.repo_id,
                    repo_type=job.repo_type,
                )
                print(f"[hf] uploaded {job.path_in_repo}")
            except Exception as e:  # noqa: BLE001
                print(f"[hf] upload failed: {e}")
            finally:
                self.q.task_done()

    def close(self, wait: bool = True) -> None:
        if wait:
            self.q.join()
        try:
            self.q.put(None, timeout=1)
        except queue.Full:
            pass
