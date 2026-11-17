# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Atomic checkpoints + optional background Hugging Face upload.

Resume is exact: model, optimizer, step counters, LR schedule, RNG state and
each rank's binpack (file, chunk, entry) cursor are stored together.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}h{(seconds % 3600) // 60:02d}m"


def checkpoint_name(
    step: int,
    elapsed_s: float,
    remaining_s: float,
    loss: float,
    upload_index: int,
    reason: str = "periodic",
) -> str:
    """Human-readable checkpoint name with run time, budget left, loss, upload #."""
    suffix = "" if reason == "periodic" else f"_{reason}"
    return (
        f"step_{step:07d}"
        f"_elapsed_{format_duration(elapsed_s)}"
        f"_left_{format_duration(remaining_s)}"
        f"_loss_{loss:.5f}"
        f"_up_{upload_index:03d}{suffix}.pt"
    )


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


def resolve_hf_token() -> str | None:
    """Token precedence: env -> Kaggle secret -> local ``hf auth login``."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            return token
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token
    except Exception:  # noqa: BLE001 - not running on Kaggle
        pass
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # noqa: BLE001
        return None


def prepare_hf(
    repo_id: str,
    repo_type: str = "model",
    token: str | None = None,
):
    """Authenticate, print the real account name, and ensure the repo exists."""
    from huggingface_hub import HfApi, login, whoami

    tok = token or resolve_hf_token()
    if not tok:
        raise RuntimeError(
            "HF upload is enabled but no token was found. Set HF_TOKEN, add a "
            "Kaggle secret named HF_TOKEN, or run `hf auth login`."
        )
    login(token=tok, add_to_git_credential=False)
    me = whoami(token=tok)
    account = str(me.get("name") or "?")
    orgs = {str(o.get("name")) for o in me.get("orgs", [])}
    owner = repo_id.split("/", 1)[0]
    print(f"[hf] logged in as {account}")
    print(f"[hf] target repo {repo_id}")
    if owner != account and owner not in orgs:
        print(
            f"[hf] WARNING: repo owner '{owner}' is neither '{account}' nor one "
            f"of its orgs {sorted(orgs)}"
        )
    api = HfApi(token=tok)
    url = api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=False)
    print(f"[hf] repo ready: {url}")
    return api, tok, account


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

    def __init__(self, token: str | None = None) -> None:
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.q: queue.Queue[_UploadJob | None] = queue.Queue(maxsize=32)
        self.thread = threading.Thread(target=self._run, daemon=True, name="hf-upload")
        self.thread.start()

    def submit(self, local: str, repo_id: str, path_in_repo: str, repo_type: str = "model") -> None:
        job = _UploadJob(local, repo_id, path_in_repo, repo_type)
        try:
            self.q.put_nowait(job)
        except queue.Full:
            print("[hf] upload queue full, skipping this checkpoint upload")

    def _run(self) -> None:
        while True:
            job = self.q.get()
            if job is None:
                return
            try:
                self.api.upload_file(
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


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def alias_latest(source: str | Path, latest: str | Path) -> None:
    latest = Path(latest)
    latest.parent.mkdir(parents=True, exist_ok=True)
    tmp = latest.with_suffix(latest.suffix + ".tmp")
    shutil.copy2(source, tmp)
    os.replace(tmp, latest)
