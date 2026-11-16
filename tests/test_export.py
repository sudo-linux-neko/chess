# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pytest
import torch

import nnue_core
from src.export import export_net, quantize_model
from src.model import NNUEModel

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "2r3k1/1p3ppp/pq2b3/3p4/3P4/P1N1P3/1P3PPP/2R2RK1 b - - 0 1",
    "4k3/8/8/8/8/8/8/R3K3 b - - 0 1",
]


def _features():
    feats, stm, buckets = [], [], []
    for fen in FENS:
        f, _, _, b = nnue_core.fen_features(fen)
        feats.append(f)
        stm.append(0 if fen.split()[1] == "w" else 1)
        buckets.append(b)
    return (
        np.array(feats, dtype=np.int16),
        np.array(stm, dtype=np.uint8),
        torch.tensor(buckets, dtype=torch.int64),
    )


def test_quantized_shapes():
    q = quantize_model(NNUEModel())
    assert q["ft_weight"].shape == (22528, 128)
    assert q["psqt"].shape == (22528, 8)
    assert q["l1_weight"].shape == (8, 16, 128)
    assert q["l2_weight"].shape == (8, 32, 32)
    assert (q["l2_weight"][:, :, 30:] == 0).all()
    assert q["out_weight"].shape == (8, 1, 32)


def test_export_matches_float(tmp_path):
    torch.manual_seed(1)
    m = NNUEModel()
    with torch.no_grad():
        m.l1_w.normal_(0.0, 0.5)
        m.l2_w.normal_(0.0, 0.5)
        m.out_w.normal_(0.0, 0.5)
    m.clip_weights_()
    m.eval()

    path = tmp_path / "net.nnue"
    meta = export_net(m, path, "unit test net")
    version, net_hash, ft_hash, _ = nnue_core.nnue_info(str(path))
    assert version == 0x7AF32F20
    assert ft_hash == 0x7F234CB8 ^ (128 * 2)
    assert net_hash == meta.net_hash

    feats, stm, bucket = _features()
    with torch.no_grad():
        pred = m(torch.tensor(feats, dtype=torch.int32), torch.tensor(stm, dtype=torch.int64), bucket)
        pred = pred.reshape(-1).numpy() * 600.0
    ints = np.asarray(nnue_core.eval_nnue(str(path), feats, stm), dtype=np.float64)
    err = np.abs(ints - pred)
    assert float(np.mean(err)) < 25.0
    assert float(np.corrcoef(ints, pred)[0, 1]) > 0.99
