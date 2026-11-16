# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stockfish 16.1 Small trainer (Rust data + PyTorch GPU training)."""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml

from .checkpoint import (
    HFUploader,
    load_checkpoint,
    save_checkpoint,
    try_download_checkpoint,
)
from .data import BinpackStream, DataConfig, StreamState, ensure_files, move_batch_to
from .export import DEFAULT_DESCRIPTION, export_net
from .loss import lambda_at, nnue_loss
from .model import NNUEModel
from .verify import verify_export


@dataclass
class TrainCfg:
    time_budget_hours: float = 9.0
    batch_size: int = 16384  # per GPU
    grad_accum: int = 1
    lr: float = 1.0e-3
    lr_min: float = 5.0e-5
    warmup_steps: int = 1000
    max_steps: int = 0  # 0 = auto-size from the time budget
    min_steps: int = 100
    measure_steps: int = 30
    log_every: int = 50
    checkpoint_minutes: float = 10.0
    compile: bool = False
    seed: int = 42
    lambda_start: float = 1.0
    lambda_end: float = 1.0
    grad_clip: float = 0.0
    loss_window: int = 100


@dataclass
class CheckpointCfg:
    dir: str = "./checkpoints"
    hf_repo: str = ""
    repo_type: str = "model"
    upload: bool = False
    resume: bool = True
    latest_name: str = "latest.pt"


@dataclass
class ExportCfg:
    enabled: bool = True
    description: str = DEFAULT_DESCRIPTION
    verify_samples: int = 4096


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainCfg = field(default_factory=TrainCfg)
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    export: ExportCfg = field(default_factory=ExportCfg)


def _build(dc, section: dict[str, Any]):
    known = set(dc.__dataclass_fields__)
    clean = {k: v for k, v in section.items() if k in known}
    return dc(**clean)


def load_config(path: str | None) -> Config:
    raw: dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    return Config(
        data=_build(DataConfig, raw.get("data", {})),
        train=_build(TrainCfg, raw.get("train", {})),
        checkpoint=_build(CheckpointCfg, raw.get("checkpoint", {})),
        export=_build(ExportCfg, raw.get("export", {})),
    )


def setup_distributed() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
    return world, rank, local


def pick_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu", local_rank)
    return torch.device("cpu")


def lr_at(step: int, total: int, cfg: TrainCfg) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    span = max(1, total - cfg.warmup_steps)
    t = min(1.0, max(0.0, (step - cfg.warmup_steps) / span))
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * t))


def _broadcast_int(value: int | None, world: int, rank: int) -> int:
    if world <= 1:
        return int(value or 0)
    box = [int(value or 0)]
    dist.broadcast_object_list(box, src=0)
    return int(box[0])


def _gather_states(local: dict[str, int], world: int, rank: int) -> list[dict[str, int]]:
    if world <= 1:
        return [local]
    out: list[Any] = [None] * world
    dist.all_gather_object(out, local)
    return out  # type: ignore[return-value]


def _unwrap(model: nn.Module) -> NNUEModel:
    return model.module if hasattr(model, "module") else model  # type: ignore[return-value]


def run(config: Config, config_path: str | None = None) -> None:
    world, rank, local_rank = setup_distributed()
    main = rank == 0
    device = pick_device(local_rank)
    torch.manual_seed(config.train.seed + rank)
    t0 = time.monotonic()

    # --- data (rank 0 downloads, everyone uses the same paths) ---
    if main:
        print(f"[init] world={world} device={device} torch={torch.__version__}")
    if main:
        paths = ensure_files(config.data)
    else:
        paths = []
    if world > 1:
        box = [paths]
        dist.broadcast_object_list(box, src=0)
        paths = box[0]
    if not paths:
        raise RuntimeError("no binpacks available")

    per_rank = config.train.batch_size
    global_batch = per_rank * world * config.train.grad_accum

    model = NNUEModel().to(device)
    if config.train.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune-no-cudagraphs")
            if main:
                print("[init] torch.compile enabled")
        except Exception as e:  # noqa: BLE001
            if main:
                print(f"[init] torch.compile failed, using eager: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, betas=(0.9, 0.999), eps=1.0e-7, weight_decay=0.0
    )

    ckpt_path = Path(config.checkpoint.dir) / config.checkpoint.latest_name
    state: dict[str, Any] | None = None
    if config.checkpoint.resume:
        if not ckpt_path.exists() and config.checkpoint.hf_repo:
            if main:
                try_download_checkpoint(
                    ckpt_path, config.checkpoint.hf_repo, config.checkpoint.latest_name,
                    config.checkpoint.repo_type,
                )
            if world > 1:
                dist.barrier()
        if ckpt_path.exists():
            state = load_checkpoint(ckpt_path, map_location="cpu")
            _unwrap(model).load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if main:
                print(f"[init] resumed step {state['step']}/{state['total_steps']} from {ckpt_path}")

    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    start_step = int(state["step"]) if state else 0
    total_steps = int(state["total_steps"]) if state and state.get("total_steps") else 0
    if config.train.max_steps > 0:
        total_steps = config.train.max_steps

    data_states = state.get("data_states") if state else None
    local_state = StreamState.from_any(data_states[rank] if data_states and len(data_states) == world else None)
    stream = BinpackStream(
        paths, world_size=world, rank=rank, state=local_state, cycle=config.data.cycle
    )

    uploader = HFUploader() if (config.checkpoint.upload and config.checkpoint.hf_repo and main) else None
    stop = {"flag": False}

    def _handle_signal(signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    def do_checkpoint(step: int, best: float, label: str | None = None) -> None:
        states = _gather_states(stream.state_dict(), world, rank)
        if not main:
            return
        path = ckpt_path if label is None else Path(config.checkpoint.dir) / label
        save_checkpoint(
            path,
            model,
            optimizer,
            step,
            total_steps,
            states,
            asdict(config),
            best_loss=best,
            extra={"config_path": config_path, "global_batch": global_batch},
        )
        print(f"[ckpt] saved {path} at step {step}")
        if uploader and label is None:
            uploader.submit(str(path), config.checkpoint.hf_repo, config.checkpoint.latest_name,
                            config.checkpoint.repo_type)

    loss_hist: deque[float] = deque(maxlen=config.train.loss_window)
    best_loss = float(state["best_loss"]) if state else float("inf")
    last_log = time.monotonic()
    last_ckpt = time.monotonic()
    measure_done = total_steps > 0
    measure_t0 = time.monotonic()
    measure_steps_done = 0

    step = start_step
    model.train()
    if main:
        print(
            f"[train] per_gpu={per_rank} global={global_batch} "
            f"start={start_step} total={total_steps or 'auto'} budget={config.train.time_budget_hours}h"
        )

    while True:
        if total_steps and step >= total_steps:
            break
        if world > 1:
            box = [bool(stop["flag"])]
            dist.broadcast_object_list(box, src=0)
            stop["flag"] = bool(box[0])
        if stop["flag"]:
            if main:
                print("[train] signal received, checkpointing")
            do_checkpoint(step, best_loss)
            break

        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.train.grad_accum):
            batch = move_batch_to(stream.next_cpu(per_rank), device)
            pred = model(batch["features"], batch["stm"], batch["bucket"])
            lam = (
                lambda_at(step, total_steps, config.train.lambda_start, config.train.lambda_end)
                if total_steps
                else config.train.lambda_start
            )
            loss = nnue_loss(pred, batch["score"], batch["result"], lam)
            (loss / config.train.grad_accum).backward()
            accum_loss += float(loss.detach()) / config.train.grad_accum

        if config.train.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        lr = lr_at(step, total_steps or step + 1, config.train)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()
        _unwrap(model).clip_weights_()

        loss_hist.append(accum_loss)
        step += 1
        measure_steps_done += 1

        # Auto-size the run once we have a real throughput measurement.
        if not measure_done and measure_steps_done >= config.train.measure_steps:
            elapsed = time.monotonic() - measure_t0
            sps = measure_steps_done / max(elapsed, 1e-9)
            remaining = config.train.time_budget_hours * 3600.0 - (time.monotonic() - t0)
            guess = step + max(config.train.min_steps, int(sps * max(0.0, remaining)))
            total_steps = _broadcast_int(guess, world, rank)
            measure_done = True
            measure_t0 = time.monotonic()
            measure_steps_done = 0
            if main:
                print(
                    f"[train] measured {sps:.2f} steps/s; auto total_steps={total_steps} "
                    f"({(total_steps - step) * global_batch:,} positions remaining)"
                )

        if step % config.train.log_every == 0 and main:
            now = time.monotonic()
            dt = max(now - last_log, 1e-9)
            pos_s = config.train.log_every * global_batch / dt
            avg = sum(loss_hist) / len(loss_hist)
            eta = (total_steps - step) * dt / config.train.log_every if total_steps else float("nan")
            cur = stream.state_dict()
            print(
                f"step {step}/{total_steps or '?'} loss {avg:.5f} lr {lr:.2e} "
                f"{pos_s/1e3:.1f}k pos/s eta {eta/60:.1f}m "
                f"file {cur['file_index']} byte {cur['byte_offset']} skip {cur['skip_entries']}",
                flush=True,
            )
            last_log = now
            if avg < best_loss:
                best_loss = avg

        if (time.monotonic() - last_ckpt) >= config.train.checkpoint_minutes * 60.0:
            do_checkpoint(step, best_loss)
            last_ckpt = time.monotonic()

    if total_steps and step >= total_steps:
        if main:
            print(f"[train] finished {step} steps")
        do_checkpoint(step, best_loss, label="final.pt")

    # --- export + verify ---
    if config.export.enabled and main:
        _unwrap(model).eval()
        out_dir = Path(config.checkpoint.dir)
        net_path = out_dir / "final.nnue"
        meta = export_net(_unwrap(model), net_path, config.export.description)
        print(f"[export] wrote {meta.path} ({meta.size/1e6:.2f} MB) hash {meta.net_hash:#x}")
        try:
            batch = stream.next_cpu(min(config.export.verify_samples, 8192))
            verify_export(_unwrap(model), str(net_path), batch)
        except Exception as e:  # noqa: BLE001
            print(f"[export] verification failed: {e}")
        if uploader and config.checkpoint.hf_repo:
            uploader.submit(str(net_path), config.checkpoint.hf_repo, "final.nnue",
                            config.checkpoint.repo_type)

    if uploader:
        uploader.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a Stockfish 16.1 Small NNUE")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config), args.config)


if __name__ == "__main__":
    main()
