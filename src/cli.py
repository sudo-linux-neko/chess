# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line entry points: train / export / verify / info / bench."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .data import BinpackStream, ensure_files, move_batch_to
from .export import DEFAULT_DESCRIPTION, export_net
from .model import NNUEModel
from .train import Config, load_config, run as train_run
from .verify import bench_reader, verify_export

ROOT = Path(__file__).resolve().parents[1]


def _maybe_spawn_kaggle(args: argparse.Namespace) -> bool:
    """Launch torch.distributed.run for 2xT4 when asked from a notebook."""
    if not getattr(args, "kaggle", False) or "LOCAL_RANK" in os.environ:
        return False
    n = torch.cuda.device_count()
    if n <= 1:
        return False
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(n),
        "--master_port",
        "29511",
        str(ROOT / "run.py"),
        "train",
        "--config",
        args.config,
    ]
    if getattr(args, "kaggle", False):
        cmd.append("--kaggle")
    print("[kaggle] launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return True


def cmd_train(args: argparse.Namespace) -> None:
    if _maybe_spawn_kaggle(args):
        return
    train_run(load_config(args.config), args.config)


def _load_model_from_checkpoint(path: str) -> NNUEModel:
    state = load_checkpoint(path)
    model = NNUEModel()
    model.load_state_dict(state["model"])
    model.eval()
    return model


def cmd_export(args: argparse.Namespace) -> None:
    model = _load_model_from_checkpoint(args.checkpoint)
    meta = export_net(model, args.out, args.description)
    print(
        f"wrote {meta.path} ({meta.size/1e6:.2f} MB) version={meta.version:#x} "
        f"net_hash={meta.net_hash:#x} ft_hash={meta.ft_hash:#x}"
    )


def cmd_info(args: argparse.Namespace) -> None:
    import nnue_core

    version, net_hash, ft_hash, desc_len = nnue_core.nnue_info(args.nnue)
    print(f"version   {version:#010x}")
    print(f"net hash  {net_hash:#010x}")
    print(f"ft hash   {ft_hash:#010x}")
    print(f"desc len  {desc_len}")
    print(f"size      {Path(args.nnue).stat().st_size:,} bytes")


def cmd_verify(args: argparse.Namespace) -> None:
    model = _load_model_from_checkpoint(args.checkpoint)
    cfg: Config = load_config(args.config) if args.config else Config()
    paths = [args.binpack] if args.binpack else ensure_files(cfg.data)
    stream = BinpackStream(paths, cycle=False)
    batch = stream.next_cpu(args.samples)
    metrics = verify_export(model, args.nnue, batch)
    if metrics["corr"] < 0.99:
        raise SystemExit(f"verification failed: corr={metrics['corr']:.5f}")


def cmd_bench(args: argparse.Namespace) -> None:
    bench_reader(args.binpack, batch=args.batch, iters=args.iters)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stockfish 16.1 Small NNUE trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train from binpacks")
    t.add_argument("--config", required=True)
    t.add_argument("--kaggle", action="store_true", help="auto-launch 2-GPU DDP")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("export", help="export a checkpoint to .nnue")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--description", default=DEFAULT_DESCRIPTION)
    e.set_defaults(func=cmd_export)

    v = sub.add_parser("verify", help="compare float model to integer .nnue")
    v.add_argument("--checkpoint", required=True)
    v.add_argument("--nnue", required=True)
    v.add_argument("--binpack")
    v.add_argument("--config")
    v.add_argument("--samples", type=int, default=4096)
    v.set_defaults(func=cmd_verify)

    i = sub.add_parser("info", help="print .nnue header")
    i.add_argument("--nnue", required=True)
    i.set_defaults(func=cmd_info)

    b = sub.add_parser("bench", help="measure binpack + feature throughput")
    b.add_argument("--binpack", required=True)
    b.add_argument("--batch", type=int, default=200_000)
    b.add_argument("--iters", type=int, default=5)
    b.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
