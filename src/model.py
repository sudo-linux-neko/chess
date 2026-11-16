# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stockfish 16.1 "Small" NNUE, exact architecture.

    HalfKAv2_hm (22528) -> FT (128 per perspective) -> pairwise product (128)
      -> 8 layer stacks selected by (piece_count-1)//4
      -> 16 -> [sqr 15 | clipped 15] -> 32 -> 1
      -> + skip + psqt[bucket]

The float forward mirrors Stockfish's integer inference (including the
127/128 pairwise scaling) so quantization in :mod:`src.export` is faithful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 22528
PAD_INDEX = INPUT_DIM
L1 = 128
L2 = 15
L3 = 32
LAYER_STACKS = 8
PSQT_BUCKETS = 8
NNUE2SCORE = 600.0
QUANTIZED_ONE = 127.0
WEIGHT_SCALE_HIDDEN = 64.0
WEIGHT_SCALE_OUT = 16.0

# Quantization limits, derived from SF16.1's integer layers.
MAX_HIDDEN_WEIGHT = QUANTIZED_ONE / WEIGHT_SCALE_HIDDEN  # 127/64
MAX_OUT_WEIGHT = (QUANTIZED_ONE * QUANTIZED_ONE) / (NNUE2SCORE * WEIGHT_SCALE_OUT)

# Stockfish's initial PSQT material values, in internal units (nnue-pytorch's
# halfka_psqts()). White pieces are positive, black pieces negative.
PSQT_PIECE_VALUES = {1: 126, 2: 781, 3: 825, 4: 1276, 5: 2538}
KING_BUCKETS = [
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10, 9, 8,
    -1, -1, -1, -1, 7, 6, 5, 4,
    -1, -1, -1, -1, 3, 2, 1, 0,
]


def _halfka_index(king_sq: int, sq: int, piece_type: int, black: bool) -> int:
    plane = (piece_type - 1) * 2 + int(black)
    if plane == 11:
        plane = 10
    kfile = king_sq & 7
    orient_king = (7 if kfile < 4 else 0) ^ king_sq
    orient_sq = (7 if kfile < 4 else 0) ^ sq
    return orient_sq + plane * 64 + KING_BUCKETS[orient_king] * 704


def initial_psqt() -> torch.Tensor:
    """[INPUT_DIM, 8] PSQT weights in ``1.0 == 600`` units."""
    out = torch.zeros(INPUT_DIM, PSQT_BUCKETS, dtype=torch.float32)
    for king_sq in range(64):
        for sq in range(64):
            for pt, val in PSQT_PIECE_VALUES.items():
                for black in (False, True):
                    idx = _halfka_index(king_sq, sq, pt, black)
                    out[idx, :] = (-val if black else val) / NNUE2SCORE
    return out


@dataclass(frozen=True)
class ModelConfig:
    l1: int = L1
    l2: int = L2
    l3: int = L3
    layer_stacks: int = LAYER_STACKS
    psqt_buckets: int = PSQT_BUCKETS


class NNUEModel(nn.Module):
    """Trainable float model whose export is a Stockfish 16.1 Small net."""

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        l1, l2, l3, ns, npsqt = cfg.l1, cfg.l2, cfg.l3, cfg.layer_stacks, cfg.psqt_buckets

        # Feature transformer: one shared table for both perspectives.
        # Row PAD_INDEX is a permanent zero row (padding_idx).
        self.ft = nn.Parameter(torch.empty(INPUT_DIM + 1, l1 + npsqt))
        self.ft_bias = nn.Parameter(torch.zeros(l1 + npsqt))

        self.l1_w = nn.Parameter(torch.empty(ns, l2 + 1, l1))
        self.l1_b = nn.Parameter(torch.empty(ns, l2 + 1))
        self.l2_w = nn.Parameter(torch.empty(ns, l3, 2 * l2))
        self.l2_b = nn.Parameter(torch.empty(ns, l3))
        self.out_w = nn.Parameter(torch.empty(ns, 1, l3))
        self.out_b = nn.Parameter(torch.empty(ns, 1))

        self._init_params()

    def _init_params(self) -> None:
        cfg = self.cfg
        l1, l2, l3, ns = cfg.l1, cfg.l2, cfg.l3, cfg.layer_stacks
        sigma = math.sqrt(1.0 / INPUT_DIM)
        with torch.no_grad():
            self.ft.uniform_(-sigma, sigma)
            self.ft[PAD_INDEX].zero_()
            self.ft[:INPUT_DIM, l1:] = initial_psqt()
            self.ft_bias.zero_()  # psqt bias must stay 0 (cancels in inference)

            # Tie all layer stacks at init; they diverge during training.
            nn.init.kaiming_uniform_(self.l1_w[0], a=math.sqrt(5))
            for i in range(1, ns):
                self.l1_w[i].copy_(self.l1_w[0])
            bound = 1.0 / math.sqrt(l1)
            self.l1_b[0].uniform_(-bound, bound)
            for i in range(1, ns):
                self.l1_b[i].copy_(self.l1_b[0])

            nn.init.kaiming_uniform_(self.l2_w[0], a=math.sqrt(5))
            for i in range(1, ns):
                self.l2_w[i].copy_(self.l2_w[0])
            bound = 1.0 / math.sqrt(2 * l2)
            self.l2_b[0].uniform_(-bound, bound)
            for i in range(1, ns):
                self.l2_b[i].copy_(self.l2_b[0])

            nn.init.kaiming_uniform_(self.out_w[0], a=math.sqrt(5))
            for i in range(1, ns):
                self.out_w[i].copy_(self.out_w[0])
            self.out_b.zero_()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        features: torch.Tensor,
        stm: torch.Tensor,
        bucket: torch.Tensor,
    ) -> torch.Tensor:
        """Return side-to-move relative evaluation in ``1.0 == 600`` units.

        features: [n, 64] int32 (white 32 | black 32, padded with 22528)
        stm:      [n] uint8/int (0 = white to move, 1 = black)
        bucket:   [n] int (0..7), (piece_count - 1) // 4
        """
        n = features.shape[0]
        flat = features.reshape(-1)
        offsets = torch.arange(
            0, 2 * n * 32, 32, dtype=torch.long, device=features.device
        )
        # 2n bags (one per perspective), each a sum of 32 rows.
        acc = F.embedding_bag(
            flat,
            self.ft,
            offsets,
            mode="sum",
            padding_idx=PAD_INDEX,
        )  # [2n, l1 + 8]
        acc = acc.view(n, 2, -1) + self.ft_bias

        l1 = self.cfg.l1
        value = acc[:, :, :l1]
        psqt = acc[:, :, l1:]

        us = (1 - stm.to(torch.float32)).unsqueeze(1)  # 1.0 when White to move
        them = 1.0 - us
        # Order perspectives side-to-move first, as Stockfish does.
        l0 = us.unsqueeze(1) * value + them.unsqueeze(1) * value.flip(1)
        l0 = l0.clamp_(0.0, 1.0)

        half = l1 // 2
        # Pairwise product, with the quantized 127/128 factor.
        pair = (l0[:, :, :half] * l0[:, :, half:]) * (127.0 / 128.0)
        x = pair.reshape(n, l1)

        # Select the layer stack per sample instead of evaluating all 8.
        w1 = self.l1_w.index_select(0, bucket)  # [n, 16, 128]
        b1 = self.l1_b.index_select(0, bucket)
        h = torch.bmm(w1, x.unsqueeze(2)).squeeze(2) + b1  # [n, 16]
        c = h[:, : self.cfg.l2]
        skip = h[:, self.cfg.l2]  # [n]
        a = torch.cat([(c * c) * (127.0 / 128.0), c], dim=1).clamp_(0.0, 1.0)

        w2 = self.l2_w.index_select(0, bucket)  # [n, 32, 30]
        b2 = self.l2_b.index_select(0, bucket)
        l2v = torch.bmm(w2, a.unsqueeze(2)).squeeze(2) + b2  # [n, 32]
        l2v = l2v.clamp_(0.0, 1.0)
        w3 = self.out_w.index_select(0, bucket)  # [n, 1, 32]
        b3 = self.out_b.index_select(0, bucket)
        positional = torch.bmm(w3, l2v.unsqueeze(2)).squeeze(2) + b3  # [n,1]
        skip_v = skip.unsqueeze(1)

        psqt_idx = bucket.view(n, 1)
        psqt_v = (psqt[:, 0, :].gather(1, psqt_idx) - psqt[:, 1, :].gather(1, psqt_idx))
        psqt_v = psqt_v * (us - 0.5)

        return positional + skip_v + psqt_v  # [n,1]

    @torch.no_grad()
    def clip_weights_(self) -> None:
        """Clamp to the exact ranges representable by Stockfish's int8 layers."""
        self.l1_w.clamp_(-MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
        self.l2_w.clamp_(-MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
        self.out_w.clamp_(-MAX_OUT_WEIGHT, MAX_OUT_WEIGHT)
        # int16 FT weights: 32767/127 is the representable bound.
        self.ft.clamp_(-32767.0 / QUANTIZED_ONE, 32767.0 / QUANTIZED_ONE)
        self.ft[PAD_INDEX].zero_()
        # PSQT bias must remain exactly zero.
        self.ft_bias[self.cfg.l1 :].zero_()


def build_model(**kwargs) -> NNUEModel:
    return NNUEModel(ModelConfig(**kwargs) if kwargs else ModelConfig())
