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
from datetime import timedelta
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml

from .checkpoint import (
    HFUploader,
    alias_latest,
    checkpoint_name,
    format_duration,
    load_checkpoint,
    prepare_hf,
    save_checkpoint,
    try_download_checkpoint,
    write_json,
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
    max_steps: int = 0  # 0 = auto-size from the time budget (recalibrates)
    min_steps: int = 100
    measure_steps: int = 200
    measure_warmup: int = 50
    log_every: int = 50
    checkpoint_minutes: float = 10.0
    compile: bool = False
    cpu_threads: int = 0  # 0 = auto; 1 is usually best with 2 DDP ranks
    seed: int = 42
    lambda_start: float = 1.0
    lambda_end: float = 1.0
    grad_clip: float = 0.0
    loss_window: int = 100


@dataclass
class CheckpointCfg:
    dir: str = "./checkpoints"
    hf_repo: str = "neko-i-am-a-cat/chess-nnue"
    repo_type: str = "model"
    upload: bool = True
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
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=60),
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
    return world, rank, local


def pick_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu", local_rank)
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


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
    """Strip DDP and torch.compile wrappers to the real NNUEModel."""
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod  # type: ignore[attr-defined]
        elif hasattr(model, "module"):
            model = model.module  # type: ignore[attr-defined]
        else:
            return model  # type: ignore[return-value]


def run(config: Config, config_path: str | None = None) -> None:
    if os.environ.get("NNUE_NO_RESUME") == "1":
        config.checkpoint.resume = False
    world, rank, local_rank = setup_distributed()
    main = rank == 0
    device = pick_device(local_rank)
    torch.manual_seed(config.train.seed + rank)
    t0 = time.monotonic()
    cpu_threads = config.train.cpu_threads
    if cpu_threads <= 0:
        cpu_threads = max(1, (os.cpu_count() or 4) // (2 * max(1, world)))
    torch.set_num_threads(cpu_threads)

    # --- Hugging Face auth/repo (default on; account name is never hardcoded) ---
    hf_token: str | None = None
    hf_account: str | None = None
    do_upload = bool(config.checkpoint.upload and config.checkpoint.hf_repo)
    if config.checkpoint.upload and not config.checkpoint.hf_repo:
        if main:
            print("[hf] upload requested but checkpoint.hf_repo is empty; disabling upload")
    if do_upload and main:
        _, hf_token, hf_account = prepare_hf(
            config.checkpoint.hf_repo, config.checkpoint.repo_type
        )
        print(f"[hf] uploading as account '{hf_account}' to {config.checkpoint.hf_repo}")
    if world > 1 and do_upload:
        box = [hf_account]
        dist.broadcast_object_list(box, src=0)
        hf_account = box[0]

    # --- data (rank 0 downloads, everyone uses the same paths) ---
    if main:
        print(
            f"[init] world={world} device={device} torch={torch.__version__} "
            f"cpu_threads={cpu_threads}"
        )
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
            from torch import _dynamo

            _dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="default")
            if main:
                print("[init] torch.compile enabled (mode=default)")
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
            try:
                state = load_checkpoint(ckpt_path, map_location="cpu")
                _unwrap(model).load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                if main:
                    print(
                        f"[init] resumed step {state['step']}/{state['total_steps']} "
                        f"from {ckpt_path}"
                    )
            except Exception as e:  # noqa: BLE001
                state = None
                if main:
                    bad = ckpt_path.with_suffix(ckpt_path.suffix + ".bad")
                    try:
                        os.replace(ckpt_path, bad)
                    except OSError:
                        pass
                    print(f"[init] resume failed ({e}); starting fresh (moved to {bad})")

    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type in ("cuda", "xpu") else None,
            broadcast_buffers=False,
        )

    start_step = int(state["step"]) if state else 0
    total_steps = int(state["total_steps"]) if state and state.get("total_steps") else 0
    if config.train.max_steps > 0:
        total_steps = config.train.max_steps

    data_states = state.get("data_states") if state else None
    local_state = StreamState.from_any(
        data_states[rank] if data_states and len(data_states) == world else None
    )
    # On a fresh run, start ranks on different files so the global stream mixes
    # all datasets from the first batch instead of walking them in lockstep.
    if local_state is None:
        local_state = StreamState()
    start_file = rank % len(paths) if len(paths) > 1 else local_state.file_index
    stream = BinpackStream(
        paths,
        world_size=world,
        rank=rank,
        state=StreamState(start_file, local_state.byte_offset, local_state.skip_entries),
        cycle=config.data.cycle,
    )

    uploader = HFUploader(token=hf_token) if (do_upload and main) else None
    state_path = Path(config.checkpoint.dir) / "training_state.json"
    loss_hist: deque[float] = deque(maxlen=config.train.loss_window)
    best_loss = float(state["best_loss"]) if state else float("inf")
    upload_index = int(state.get("upload_index", 0)) if state else 0
    stop = {"flag": False}

    def _handle_signal(signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    def do_checkpoint(
        step: int,
        best: float,
        label: str | None = None,
        reason: str = "periodic",
        do_upload: bool = True,
    ) -> None:
        nonlocal upload_index
        states = _gather_states(stream.state_dict(), world, rank)
        if not main:
            # Keep non-main ranks from racing into the next DDP all-reduce while
            # rank 0 is still writing the checkpoint.
            if world > 1:
                dist.barrier()
            return
        now = time.monotonic()
        elapsed = now - t0
        remaining = max(0.0, config.train.time_budget_hours * 3600.0 - elapsed)
        cur_loss = sum(loss_hist) / len(loss_hist) if loss_hist else 0.0
        next_index = upload_index + 1 if (uploader and do_upload) else upload_index
        if label is None:
            name = checkpoint_name(step, elapsed, remaining, cur_loss, next_index, reason)
        else:
            name = label
        path = Path(config.checkpoint.dir) / name
        save_checkpoint(
            path,
            model,
            optimizer,
            step,
            total_steps,
            states,
            asdict(config),
            best_loss=best,
            extra={
                "config_path": config_path,
                "global_batch": global_batch,
                "upload_index": next_index,
                "elapsed_s": elapsed,
                "remaining_s": remaining,
                "loss": cur_loss,
                "reason": reason,
                "hf_account": hf_account,
                "hf_repo": config.checkpoint.hf_repo,
            },
        )
        alias_latest(path, ckpt_path)
        if uploader and do_upload:
            upload_index = next_index
            remote = f"checkpoints/{name}" if label is None else name
            uploader.submit(str(path), config.checkpoint.hf_repo, remote,
                            config.checkpoint.repo_type)
            uploader.submit(str(ckpt_path), config.checkpoint.hf_repo,
                            config.checkpoint.latest_name, config.checkpoint.repo_type)
            write_json(
                state_path,
                {
                    "account": hf_account,
                    "repo": config.checkpoint.hf_repo,
                    "repo_url": f"https://huggingface.co/{config.checkpoint.hf_repo}",
                    "name": name,
                    "step": step,
                    "total_steps": total_steps,
                    "loss": cur_loss,
                    "elapsed_s": elapsed,
                    "remaining_s": remaining,
                    "upload_index": upload_index,
                    "reason": reason,
                    "updated": int(time.time()),
                },
            )
            uploader.submit(str(state_path), config.checkpoint.hf_repo,
                            "training_state.json", config.checkpoint.repo_type)
        print(
            f"[ckpt] {name} step={step} loss={cur_loss:.5f} "
            f"elapsed={format_duration(elapsed)} left={format_duration(remaining)} "
            f"upload #{upload_index}"
        )
        if world > 1:
            dist.barrier()
    auto_steps = config.train.max_steps <= 0
    budget_seconds = max(0.0, config.train.time_budget_hours * 3600.0)
    budget_stop = max(1.0, budget_seconds - min(60.0, budget_seconds * 0.05)) if budget_seconds else 0.0
    last_log = time.monotonic()
    last_ckpt = time.monotonic()
    measure_done = not auto_steps
    measure_t0 = time.monotonic()
    measure_steps_done = 0
    measure_warmup_done = 0
    last_recal_t = time.monotonic()
    last_recal_step = start_step

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
        budget_hit = budget_stop > 0 and (time.monotonic() - t0) >= budget_stop
        stop_now = bool(stop["flag"] or budget_hit)
        ckpt_now = (time.monotonic() - last_ckpt) >= config.train.checkpoint_minutes * 60.0
        if world > 1:
            # Rank 0 owns both decisions. Checking local clocks independently made
            # one rank enter checkpoint collectives while the other was still in
            # the DDP all-reduce, which deadlocked after ~30 minutes.
            box = [[stop_now if main else False, ckpt_now if main else False]]
            dist.broadcast_object_list(box, src=0)
            stop_now, ckpt_now = bool(box[0][0]), bool(box[0][1])
        if stop_now:
            reason = "time_budget" if budget_hit and not stop["flag"] else "interrupted"
            if main:
                print(f"[train] {reason}, checkpointing")
            do_checkpoint(step, best_loss, reason=reason)
            break
        if ckpt_now:
            do_checkpoint(step, best_loss)
            last_ckpt = time.monotonic()

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
        if total_steps:
            lr = lr_at(step, total_steps, config.train)
        elif step < config.train.warmup_steps:
            lr = config.train.lr * (step + 1) / config.train.warmup_steps
        else:
            lr = config.train.lr
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()
        _unwrap(model).clip_weights_()

        loss_hist.append(accum_loss)
        step += 1

        # Auto-size once after a real warmup, then extend if throughput rises.
        if not measure_done:
            if measure_warmup_done < config.train.measure_warmup:
                measure_warmup_done += 1
            elif measure_steps_done == 0:
                _sync(device)
                measure_t0 = time.monotonic()
                measure_steps_done = 1
            else:
                measure_steps_done += 1
            if measure_steps_done >= config.train.measure_steps:
                _sync(device)
                elapsed = time.monotonic() - measure_t0
                sps = measure_steps_done / max(elapsed, 1e-9)
                remaining = budget_seconds - (time.monotonic() - t0)
                guess = step + max(config.train.min_steps, int(sps * max(0.0, remaining)))
                old = total_steps
                total_steps = _broadcast_int(guess, world, rank)
                measure_done = True
                last_recal_t = time.monotonic()
                last_recal_step = step
                if main:
                    print(
                        f"[train] measured {sps:.2f} steps/s after warmup; "
                        f"auto total_steps={total_steps} (was {old or 'auto'}) "
                        f"({(total_steps - step) * global_batch:,} positions remaining)"
                    )

        if measure_done and auto_steps and step - last_recal_step >= 5000:
            now = time.monotonic()
            dt = now - last_recal_t
            ds = step - last_recal_step
            last_recal_t = now
            last_recal_step = step
            if dt > 30.0:
                sps = ds / dt
                remaining = budget_seconds - (now - t0)
                target = step + int(sps * max(0.0, remaining))
                if target > total_steps + 5000:
                    total_steps = _broadcast_int(target, world, rank)
                    if main:
                        print(
                            f"[train] extended total_steps to {total_steps} "
                            f"({sps:.2f} steps/s, {max(0.0, remaining)/60:.0f}m left)"
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

    if main:
        print(f"[train] stopped at step {step}")
    do_checkpoint(step, best_loss, label="final.pt", reason="final")

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
