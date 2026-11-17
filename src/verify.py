# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify that an exported .nnue is loadable and numerically faithful."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

import nnue_core
from .model import NNUE2SCORE


@torch.no_grad()
def verify_export(
    model: nn.Module,
    net_path: str,
    batch: dict[str, torch.Tensor],
    verbose: bool = True,
) -> dict[str, float]:
    """Compare float predictions to Stockfish's exact integer inference."""
    if hasattr(model, "module"):
        model = model.module
    model.eval()
    device = next(model.parameters()).device
    b = {k: v.to(device) for k, v in batch.items()}
    pred = model(b["features"], b["stm"], b["bucket"]).reshape(-1) * NNUE2SCORE

    feats = b["features"].detach().cpu().numpy().astype(np.int16, copy=False)
    stm = b["stm"].detach().cpu().numpy().astype(np.uint8, copy=False)
    ints = np.asarray(nnue_core.eval_nnue(net_path, feats, stm), dtype=np.float32)

    p = pred.detach().cpu().numpy()
    err = ints - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err.astype(np.float64) ** 2)))
    corr = float(np.corrcoef(p, ints)[0, 1]) if p.std() > 1e-6 else 1.0
    if verbose:
        print(
            f"[verify] n={len(p)} MAE={mae:.2f} RMSE={rmse:.2f} max={np.abs(err).max():.0f} "
            f"corr={corr:.6f}"
        )
        print(f"[verify] float[:4]={p[:4].round(1).tolist()} int[:4]={ints[:4].round(1).tolist()}")
    return {"mae": mae, "rmse": rmse, "max": float(np.abs(err).max()), "corr": corr}


def bench_reader(path: str, batch: int = 200_000, iters: int = 5) -> float:
    r = nnue_core.BinpackReader(path)
    for _ in range(2):
        r.next_batch(batch)
    n = 0
    t = time.perf_counter()
    for _ in range(iters):
        n += r.next_batch(batch)["count"]
    dt = time.perf_counter() - t
    rate = n / dt / 1e6
    print(f"[bench] {n:,} positions in {dt:.3f}s = {rate:.2f} Mpos/s")
    return rate
