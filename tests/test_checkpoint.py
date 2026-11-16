# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
import torch

from src.checkpoint import atomic_save, load_checkpoint, save_checkpoint
from src.model import NNUEModel


def test_roundtrip(tmp_path):
    m = NNUEModel()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    path = tmp_path / "latest.pt"
    states = [{"file_index": 0, "byte_offset": 1234, "skip_entries": 7}]
    save_checkpoint(path, m, opt, 42, 100, states, {"x": 1}, best_loss=0.5)
    ckpt = load_checkpoint(path)
    assert ckpt["step"] == 42 and ckpt["total_steps"] == 100
    assert ckpt["data_states"] == states
    assert ckpt["author"] == "Vaibhav"
    m2 = NNUEModel()
    m2.load_state_dict(ckpt["model"])
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)


def test_atomic_save(tmp_path):
    p = tmp_path / "a.bin"
    atomic_save({"ok": True}, p)
    assert p.exists() and not (tmp_path / "a.bin.tmp").exists()
    assert torch.load(p, weights_only=False)["ok"] is True
