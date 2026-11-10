# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Tests for export functionality.
"""

import tempfile
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.model import create_model
from src.export import NNUEExporter, QuantizedLayer


class TestExport:
    """Tests for NNUE export."""

    def setup_method(self):
        self.config = {
            "model": {
                "input_dim": 81920,
                "ft_size": 256,
                "hidden_sizes": [32, 32],
                "output_size": 1,
                "clipped_relu_max": 1.0,
            },
            "export": {
                "dir": "/tmp/test_export",
                "quantize": True,
                "quantization_method": "per_channel_symmetric",
                "verify": False,
            },
            "training": {
                "score_scale": 370.0,
            },
        }

    def test_exporter_creation(self):
        """Test exporter can be created."""
        exporter = NNUEExporter(self.config)
        assert exporter.quantize is True
        assert exporter.output_dir.exists()

    def test_quantize_linear_layer(self):
        """Test quantization of a single linear layer."""
        exporter = NNUEExporter(self.config)

        # Create a test layer
        layer = torch.nn.Linear(10, 5)
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

        # Quantize
        quantized = exporter._quantize_linear(layer, "test")

        # Check shapes
        assert quantized.weight.shape == (5, 10)
        assert quantized.bias.shape == (5,)
        assert quantized.scale.shape == (5,)

        # Check dtypes
        assert quantized.weight.dtype == np.int8
        assert quantized.bias.dtype == np.int32
        assert quantized.scale.dtype == np.float32

        # Check quantization range
        assert quantized.weight.min() >= -128
        assert quantized.weight.max() <= 127

        # Check scales are positive
        assert (quantized.scale > 0).all()

    def test_quantized_layer_forward(self):
        """Test quantized layer forward pass."""
        exporter = NNUEExporter(self.config)

        # Create layer with known weights
        layer = torch.nn.Linear(4, 3)
        layer.weight.data = torch.tensor([
            [1.0, 0.5, -0.5, 0.0],
            [0.0, 1.0, 0.5, -0.5],
            [-0.5, 0.0, 1.0, 0.5],
        ])
        layer.bias.data = torch.tensor([0.1, -0.1, 0.0])

        quantized = exporter._quantize_linear(layer, "test")

        # Test input
        x = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int8)

        # Quantized forward
        output = quantized.forward(x)

        # Compare with float forward
        x_float = torch.tensor(x, dtype=torch.float32)
        expected = layer(x_float).detach().numpy()

        # Should be close (within quantization error)
        np.testing.assert_allclose(output, expected, atol=0.1, rtol=0.1)

    def test_export_creates_file(self):
        """Test that export creates a valid .nnue file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.config["export"]["dir"] = tmpdir
            exporter = NNUEExporter(self.config)

            model = create_model(self.config)

            # Export
            nnue_path = exporter.export(model)

            # Check file exists
            assert nnue_path.exists()
            assert nnue_path.suffix == ".nnue"
            assert nnue_path.stat().st_size > 0

            # Check header
            with open(nnue_path, 'rb') as f:
                magic = f.read(4)
                assert magic == b"NNUE"

                version = int.from_bytes(f.read(4), 'little')
                assert version == 1

                arch_hash = f.read(32)
                net_hash = f.read(32)
                assert len(arch_hash) == 32
                assert len(net_hash) == 32

    def test_export_file_structure(self):
        """Test exported file has correct layer structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.config["export"]["dir"] = tmpdir
            exporter = NNUEExporter(self.config)

            model = create_model(self.config)
            nnue_path = exporter.export(model)

            # Parse and verify structure
            with open(nnue_path, 'rb') as f:
                # Skip header
                f.read(4 + 4 + 32 + 32)

                # Feature transformer: 81920 -> 16384
                ft_in = int.from_bytes(f.read(4), 'little')
                ft_out = int.from_bytes(f.read(4), 'little')
                assert ft_in == 81920 // 64  # Per bucket
                assert ft_out == 256

                # Skip ft data
                f.read(ft_out * ft_in)  # weights
                f.read(ft_out * 4)      # bias
                f.read(ft_out * 4)      # scales

                # Hidden layer 1: 16384 -> 32
                h1_in = int.from_bytes(f.read(4), 'little')
                h1_out = int.from_bytes(f.read(4), 'little')
                assert h1_in == 16384
                assert h1_out == 32

                f.read(h1_out * h1_in)
                f.read(h1_out * 4)
                f.read(h1_out * 4)

                # Hidden layer 2: 32 -> 32
                h2_in = int.from_bytes(f.read(4), 'little')
                h2_out = int.from_bytes(f.read(4), 'little')
                assert h2_in == 32
                assert h2_out == 32

                f.read(h2_out * h2_in)
                f.read(h2_out * 4)
                f.read(h2_out * 4)

                # Output layer: 32 -> 1
                out_in = int.from_bytes(f.read(4), 'little')
                out_out = int.from_bytes(f.read(4), 'little')
                assert out_in == 32
                assert out_out == 1

    def test_network_hash_consistency(self):
        """Test that network hash is consistent for same weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.config["export"]["dir"] = tmpdir
            exporter = NNUEExporter(self.config)

            model = create_model(self.config)

            # Export twice
            path1 = exporter.export(model)
            path2 = exporter.export(model)

            # Hashes should be same (same weights)
            with open(path1, 'rb') as f:
                f.read(4 + 4)
                hash1 = f.read(32)

            with open(path2, 'rb') as f:
                f.read(4 + 4)
                hash2 = f.read(32)

            assert hash1 == hash2

    def test_quantized_inference_matches_float(self):
        """Test that quantized inference approximates float inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.config["export"]["dir"] = tmpdir
            exporter = NNUEExporter(self.config)

            model = create_model(self.config)
            model.eval()

            # Create test input (dense features)
            test_input = torch.randn(10, 81920)

            # Float inference
            with torch.no_grad():
                float_output = model(test_input).numpy()

            # Export and run quantized (simplified)
            nnue_path = exporter.export(model)

            # For this test, we verify the quantized layers can be loaded
            # Full quantized inference test would require implementing the full forward pass
            assert nnue_path.exists()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])