# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Smoke test for the entire pipeline.
Run with: python -m pytest tests/test_smoke.py -v -x
"""

import tempfile
import torch
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.model import create_model
from src.data import compile_cpp_loader, StreamingBinpackDataset
from src.train import NNUELightningModule
from src.checkpoint import CheckpointManager
from src.hf_upload import FakeHFUploader
from src.export import NNUEExporter


class TestSmokePipeline:
    """End-to-end smoke test."""

    def setup_method(self):
        """Set up minimal config for smoke test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.temp_dir.name)

        self.config = {
            "hardware": {
                "gpus": 0,  # CPU only for testing
                "precision": "32-true",
                "num_workers": 0,
                "prefetch_factor": 2,
            },
            "model": {
                "input_dim": 128,  # Small for testing
                "ft_size": 16,
                "hidden_sizes": [8, 8],
                "output_size": 1,
                "clipped_relu_max": 1.0,
            },
            "training": {
                "batch_size": 32,
                "max_positions": 100,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "wdl_lambda": 0.5,
                "score_scale": 370.0,
                "optimizer": "adamw",
                "warmup_ratio": 0.1,
                "min_lr_ratio": 0.1,
            },
            "time_budget": {
                "hours": 0.01,  # ~36 seconds
                "setup_minutes": 0,
                "checkpoint_interval_min": 1,
                "final_save_min": 0,
            },
            "data": {
                "data_dir": str(self.tmpdir / "data"),
                "max_data_gb": 1,
                "shuffle_buffer_mb": 1,
                "num_files_simultaneous": 1,
                "sources": [
                    {"type": "local", "path": str(self.tmpdir / "data")}
                ],
            },
            "checkpoint": {
                "dir": str(self.tmpdir / "checkpoints"),
                "keep_last_n": 2,
                "atomic_write": True,
                "include_optimizer": True,
                "include_scheduler": True,
                "include_scaler": False,
                "include_rng": True,
                "include_data_position": True,
            },
            "hf": {
                "enabled": True,
                "repo_id": "test/repo",
                "token_env": "HF_TOKEN",
                "auto_create_repo": False,
                "repo_private": True,
                "upload_workers": 1,
                "max_retries": 1,
                "retry_backoff_base": 0.1,
                "upload_logs": False,
                "upload_exports": False,
            },
            "export": {
                "dir": str(self.tmpdir / "export"),
                "quantize": True,
                "quantization_method": "per_channel_symmetric",
                "verify": False,
            },
            "logging": {
                "log_every_n_steps": 10,
                "log_dir": str(self.tmpdir / "logs"),
                "csv_log": True,
                "tensorboard": False,
                "log_gpu_stats": False,
            },
            "smoke_test": {
                "enabled": True,
                "max_positions": 100,
                "batch_size": 32,
                "time_budget_hours": 0.01,
                "checkpoint_interval_min": 1,
                "num_workers": 0,
                "fake_hf_upload": True,
            },
        }

        # Create dummy data
        self._create_dummy_data()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _create_dummy_data(self):
        """Create minimal dummy binpack file."""
        import struct
        data_dir = Path(self.config["data"]["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)

        binpack_path = data_dir / "smoke_test.binpack"
        num_positions = 100
        features_per_pos = 16

        with open(binpack_path, 'wb') as f:
            f.write(b"BINP")
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<Q", num_positions))

            for _ in range(num_positions):
                f.write(struct.pack("<H", features_per_pos))
                f.write(struct.pack(f"<{features_per_pos}H", *range(features_per_pos)))
                f.write(struct.pack(f"<{features_per_pos}f", *[1.0] * features_per_pos))
                f.write(struct.pack("<3f", 0.3, 0.4, 0.3))
                f.write(struct.pack("<f", 0.0))
                f.write(struct.pack("<b", 0))

    def test_model_creation(self):
        """Test model can be created."""
        model = create_model(self.config)
        assert model is not None
        assert model.get_num_parameters() > 0
        print(f"Model params: {model.get_num_parameters():,}")

    def test_forward_pass(self):
        """Test model forward pass."""
        model = create_model(self.config)
        model.eval()

        # Create dummy input matching model input_dim
        batch_size = 4
        input_dim = self.config["model"]["input_dim"]
        x = torch.randint(0, 2, (batch_size, input_dim), dtype=torch.float32)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()

    def test_checkpoint_save_load(self):
        """Test checkpoint save and load cycle."""
        model = create_model(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        class TrainingState:
            step = 42
            epoch = 1
            wall_time_elapsed = 123.45
            best_val_loss = 0.5
            data_position = {"file_idx": 0, "position": 50}

        training_state = TrainingState()

        checkpoint_manager = CheckpointManager(self.config)

        # Save
        ckpt_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            training_state=training_state,
            config=self.config,
        )

        assert ckpt_path is not None
        assert ckpt_path.exists()

        # Load into new model
        new_model = create_model(self.config)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)

        loaded_state = checkpoint_manager.load_checkpoint(
            ckpt_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            scaler=None,
        )

        assert loaded_state["step"] == 42
        assert loaded_state["epoch"] == 1
        assert abs(loaded_state["training_state"]["wall_time_elapsed"] - 123.45) < 1.0

        # Verify weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_fake_hf_upload(self):
        """Test fake HF uploader."""
        uploader = FakeHFUploader(self.config)

        # Create a dummy checkpoint
        ckpt_path = self.tmpdir / "checkpoints" / "step_0000000042.ckpt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_bytes(b"dummy checkpoint data")

        # Queue upload
        uploader.queue_upload(ckpt_path)
        uploader.queue_export(ckpt_path)

        # Shutdown
        uploader.shutdown(wait=True)

        assert len(uploader._queue) == 2

    def test_export(self):
        """Test model export to .nnue."""
        model = create_model(self.config)
        model.eval()

        exporter = NNUEExporter(self.config)

        # Create test data
        test_data = torch.randn(10, self.config["model"]["input_dim"])

        # Export
        nnue_path = exporter.export(model, calibration_data=test_data)

        assert nnue_path.exists()
        assert nnue_path.suffix == ".nnue"
        assert nnue_path.stat().st_size > 0

        # Verify header
        with open(nnue_path, 'rb') as f:
            magic = f.read(4)
            assert magic == b"NNUE"

    def test_data_loader_creation(self):
        """Test data loader can be created."""
        from src.data import create_dataloader

        loader = create_dataloader(self.config, is_train=True)
        assert loader is not None

        # Try to get one batch
        batch = next(iter(loader))
        assert "features" in batch
        assert "eval" in batch
        assert "wdl" in batch

    def test_lightning_module_creation(self):
        """Test Lightning module can be created."""
        checkpoint_manager = CheckpointManager(self.config)
        hf_uploader = FakeHFUploader(self.config)

        pl_module = NNUELightningModule(self.config, checkpoint_manager, hf_uploader)

        assert pl_module is not None
        assert pl_module.model is not None

        # Test forward
        batch_size = 2
        input_dim = self.config["model"]["input_dim"]
        batch = {
            "features": torch.randn(batch_size, input_dim),
            "weights": torch.ones(batch_size, input_dim),
            "wdl": torch.tensor([[0.3, 0.4, 0.3]] * batch_size),
            "eval": torch.zeros(batch_size),
        }

        output = pl_module(batch["features"])
        assert output.shape == (batch_size, 1)

    def test_optimizer_creation(self):
        """Test optimizer configuration."""
        checkpoint_manager = CheckpointManager(self.config)
        hf_uploader = FakeHFUploader(self.config)

        pl_module = NNUELightningModule(self.config, checkpoint_manager, hf_uploader)

        optimizers = pl_module.configure_optimizers()
        assert "optimizer" in optimizers
        assert "lr_scheduler" in optimizers

        optimizer = optimizers["optimizer"]
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_full_training_step(self):
        """Test a single training step works."""
        checkpoint_manager = CheckpointManager(self.config)
        hf_uploader = FakeHFUploader(self.config)

        pl_module = NNUELightningModule(self.config, checkpoint_manager, hf_uploader)

        batch_size = 4
        input_dim = self.config["model"]["input_dim"]
        batch = {
            "features": torch.randn(batch_size, input_dim),
            "weights": torch.ones(batch_size, input_dim),
            "wdl": torch.tensor([[0.3, 0.4, 0.3]] * batch_size),
            "eval": torch.randn(batch_size) * 100,
        }

        # This should not raise
        loss = pl_module.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)

    def test_config_hash(self):
        """Test config hash computation."""
        checkpoint_manager = CheckpointManager(self.config)

        hash1 = checkpoint_manager._compute_config_hash(self.config)
        hash2 = checkpoint_manager._compute_config_hash(self.config)

        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated SHA256


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])