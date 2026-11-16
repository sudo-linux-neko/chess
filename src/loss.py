# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Win-rate / score blend used by the current Stockfish trainer.

Kept numerically identical to nnue-pytorch's ``step_`` for SF16.1:
    scorenet = net * 600
    qf = 0.5 * (sigmoid((scorenet-270)/340) - sigmoid((-scorenet-270)/340) + 1)
    pf = same for the target score (out_scaling=380)
    pt = lambda*pf + (1-lambda)*result
    loss = mean(|pt - qf|^2.5)
"""

from __future__ import annotations

import torch

NNUE2SCORE = 600.0
IN_SCALING = 340.0
OUT_SCALING = 380.0
OFFSET = 270.0


def win_rate(x: torch.Tensor, scaling: float) -> torch.Tensor:
    q = (x - OFFSET) / scaling
    qm = (-x - OFFSET) / scaling
    return 0.5 * (1.0 + torch.sigmoid(q) - torch.sigmoid(qm))


def nnue_loss(
    pred: torch.Tensor,
    score: torch.Tensor,
    result: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """pred: [n,1] net output (1.0 == 600); score/result relative to stm."""
    scorenet = pred.reshape(-1) * NNUE2SCORE
    qf = win_rate(scorenet, IN_SCALING)
    pf = win_rate(score, OUT_SCALING)
    pt = pf * lam + result * (1.0 - lam)
    return torch.pow(torch.abs(pt - qf), 2.5).mean()


def lambda_at(step: int, total: int, start: float, end: float) -> float:
    if total <= 0:
        return end
    t = min(1.0, step / total)
    return start + (end - start) * t
