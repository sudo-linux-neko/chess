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


def test_checkpoint_name_contains_run_stats():
    from src.checkpoint import checkpoint_name, format_duration

    assert format_duration(0) == "00h00m"
    assert format_duration(3661) == "01h01m"
    name = checkpoint_name(1234, 2 * 3600 + 15 * 60, 6 * 3600 + 45 * 60, 0.02413, 7)
    assert name == "step_0001234_elapsed_02h15m_left_06h45m_loss_0.02413_up_007.pt"
    assert checkpoint_name(5, 0, 0, 1.0, 1, "interrupted").endswith("_up_001_interrupted.pt")


def test_train_config_uploads_by_default():
    from src.train import load_config

    cfg = load_config("configs/train.yaml")
    assert cfg.checkpoint.hf_repo == "neko-i-am-a-cat/chess-nnue"
    assert cfg.checkpoint.upload is True


def test_kaggle_train_config_uses_two_packs_and_auto_size():
    from src.train import load_config

    cfg = load_config("configs/train.yaml")
    assert len(cfg.data.files) == 2
    assert cfg.data.files[0].startswith("fishpack32")
    assert cfg.data.files[1].startswith("wrongIsRight")
    assert cfg.train.max_steps == 0
    assert cfg.train.measure_steps >= 100
    assert cfg.train.compile is True
