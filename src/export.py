# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Quantize a float NNUEModel into a real Stockfish 16.1 Small .nnue file.

Scales are the ones Stockfish's integer layers actually implement:
    FT weight/bias      * 127          -> int16
    FT psqt             * 600 * 16     -> int32
    hidden fc weight    * 64           -> int8   (clamp +/-127/64)
    hidden fc bias      * 64 * 127     -> int32
    output fc weight    * 600*16/127   -> int8   (clamp +/-127*127/9600)
    output fc bias      * 600 * 16     -> int32

Serialization itself is done in Rust (``nnue_core.write_nnue``) so the
LEB128 payload for ~2.9M FT weights is generated quickly and exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import nnue_core
from .model import (
    INPUT_DIM,
    MAX_HIDDEN_WEIGHT,
    MAX_OUT_WEIGHT,
    NNUE2SCORE,
    QUANTIZED_ONE,
    WEIGHT_SCALE_HIDDEN,
    WEIGHT_SCALE_OUT,
    NNUEModel,
)

DEFAULT_DESCRIPTION = "Trained by Vaibhav with the Rust+PyTorch SF16.1 trainer"

K_WEIGHT_SCALE_OUT = NNUE2SCORE * WEIGHT_SCALE_OUT / QUANTIZED_ONE  # 9600/127
K_BIAS_SCALE_OUT = NNUE2SCORE * WEIGHT_SCALE_OUT  # 9600
K_BIAS_SCALE_HIDDEN = WEIGHT_SCALE_HIDDEN * QUANTIZED_ONE  # 8128


def _q(t: torch.Tensor, scale: float, dtype: torch.dtype, lo=None, hi=None) -> np.ndarray:
    x = t.detach().to(torch.float32)
    if lo is not None or hi is not None:
        x = x.clamp(lo, hi)
    x = torch.round(x * scale)
    info = torch.iinfo(dtype)
    x = x.clamp(info.min, info.max).to(dtype)
    return x.contiguous().cpu().numpy()


def unwrap(model: nn.Module) -> NNUEModel:
    return model.module if hasattr(model, "module") else model  # type: ignore[return-value]


@dataclass(frozen=True)
class ExportMeta:
    path: str
    size: int
    version: int
    net_hash: int
    ft_hash: int
    description: str


def quantize_model(model: nn.Module) -> dict[str, np.ndarray]:
    m = unwrap(model)
    cfg = m.cfg
    l1, l2, l3, ns = cfg.l1, cfg.l2, cfg.l3, cfg.layer_stacks

    ft_w = _q(m.ft[:INPUT_DIM, :l1], QUANTIZED_ONE, torch.int16)
    ft_b = _q(m.ft_bias[:l1], QUANTIZED_ONE, torch.int16)
    psqt = _q(m.ft[:INPUT_DIM, l1:], NNUE2SCORE * WEIGHT_SCALE_OUT, torch.int32)

    l1_w = _q(m.l1_w, WEIGHT_SCALE_HIDDEN, torch.int8, -MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
    l1_b = _q(m.l1_b, K_BIAS_SCALE_HIDDEN, torch.int32)
    l2_w = torch.zeros(ns, l3, 32, dtype=m.l2_w.dtype, device=m.l2_w.device)
    l2_w[:, :, : 2 * l2] = m.l2_w.detach()
    l2_w = _q(l2_w, WEIGHT_SCALE_HIDDEN, torch.int8, -MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
    l2_b = _q(m.l2_b, K_BIAS_SCALE_HIDDEN, torch.int32)

    out_w = _q(m.out_w, K_WEIGHT_SCALE_OUT, torch.int8, -MAX_OUT_WEIGHT, MAX_OUT_WEIGHT)
    out_b = _q(m.out_b, K_BIAS_SCALE_OUT, torch.int32)

    return {
        "ft_bias": ft_b,
        "ft_weight": ft_w,
        "psqt": psqt,
        "l1_bias": l1_b,
        "l1_weight": l1_w,
        "l2_bias": l2_b,
        "l2_weight": l2_w,
        "out_bias": out_b,
        "out_weight": out_w,
    }


def export_net(
    model: nn.Module,
    path: str | Path,
    description: str = DEFAULT_DESCRIPTION,
) -> ExportMeta:
    q = quantize_model(model)
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nnue_core.write_nnue(
        path,
        description,
        q["ft_bias"],
        q["ft_weight"],
        q["psqt"],
        q["l1_bias"],
        q["l1_weight"],
        q["l2_bias"],
        q["l2_weight"],
        q["out_bias"],
        q["out_weight"],
    )
    version, net_hash, ft_hash, _ = nnue_core.nnue_info(path)
    return ExportMeta(
        path=path,
        size=Path(path).stat().st_size,
        version=version,
        net_hash=net_hash,
        ft_hash=ft_hash,
        description=description,
    )
