# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
import math

import torch

from src.loss import lambda_at, nnue_loss, win_rate


def test_win_rate_symmetry():
    x = torch.tensor([-500.0, -100.0, 0.0, 100.0, 500.0])
    w = win_rate(x, 340.0)
    assert torch.allclose(w, 1.0 - win_rate(-x, 340.0), atol=1e-6)
    assert abs(float(w[2]) - 0.5) < 1e-6


def test_loss_is_positive_and_differentiable():
    pred = torch.zeros(4, 1, requires_grad=True)
    score = torch.tensor([0.0, 100.0, -300.0, 50.0])
    result = torch.tensor([0.0, 1.0, -1.0, 0.0])
    loss = nnue_loss(pred, score, result, 1.0)
    assert loss.item() > 0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_lambda_schedule():
    assert lambda_at(0, 100, 1.0, 0.5) == 1.0
    assert abs(lambda_at(100, 100, 1.0, 0.5) - 0.5) < 1e-9
    assert math.isclose(lambda_at(50, 0, 1.0, 0.0), 0.0)
