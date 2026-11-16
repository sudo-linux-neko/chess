# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
import torch

import nnue_core
from src.model import MAX_HIDDEN_WEIGHT, MAX_OUT_WEIGHT, NNUEModel, initial_psqt


def _batch(n=8):
    feats, stm, bucket = [], [], []
    for fen, side in (
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3", 1),
    ):
        f, _, _, b = nnue_core.fen_features(fen)
        for _ in range(n // 2):
            feats.append(f)
            stm.append(side)
            bucket.append(b)
    return (
        torch.tensor(feats, dtype=torch.int32),
        torch.tensor(stm, dtype=torch.int64),
        torch.tensor(bucket, dtype=torch.int64),
    )


def test_initial_psqt_material_values():
    p = initial_psqt()
    assert p.shape == (22528, 8)
    assert p.abs().max().item() > 2.0
    assert p[22528 - 1].eq(0).all()


def test_forward_shape_and_grad():
    torch.manual_seed(0)
    m = NNUEModel()
    x, stm, bucket = _batch()
    out = m(x, stm, bucket)
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert m.ft.grad is not None and m.l1_w.grad is not None
    assert m.ft.grad[22528].abs().sum() == 0


def test_clip_weights_bounds():
    m = NNUEModel()
    with torch.no_grad():
        m.l1_w.fill_(10.0)
        m.l2_w.fill_(10.0)
        m.out_w.fill_(10.0)
        m.ft.fill_(1000.0)
    m.clip_weights_()
    assert m.l1_w.abs().max() <= MAX_HIDDEN_WEIGHT + 1e-6
    assert m.l2_w.abs().max() <= MAX_HIDDEN_WEIGHT + 1e-6
    assert m.out_w.abs().max() <= MAX_OUT_WEIGHT + 1e-6
    assert m.ft[22528].eq(0).all()
