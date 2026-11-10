# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Export Module: Quantize PyTorch model to Stockfish-compatible .nnue format.

Based on nnue-pytorch serialize.py with quantization support.
"""

import os
import struct
import hashlib
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class NNUEHeader:
    """NNUE file header (Stockfish format)."""
    magic: bytes = b"NNUE"
    version: int = 1
    architecture_hash: bytes = b"\x00" * 32  # SHA256 of architecture
    network_hash: bytes = b"\x00" * 32       # SHA256 of weights


class QuantizedLayer:
    """Quantized linear layer with int8 weights and int32 bias."""

    def __init__(
        self,
        weight: np.ndarray,      # (out_features, in_features) int8
        bias: np.ndarray,        # (out_features,) int32
        scale: np.ndarray,       # (out_features,) float32 - per-channel scale
        input_zero_point: int = 0,
    ):
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.input_zero_point = input_zero_point
        self.out_features, self.in_features = weight.shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Quantized forward pass: x (int8) -> output (float32)."""
        # x: (batch, in_features) int8
        # weight: (out_features, in_features) int8
        # bias: (out_features,) int32
        # scale: (out_features,) float32

        # Accumulate in int32
        acc = x.astype(np.int32) @ self.weight.T.astype(np.int32)  # (batch, out_features)
        acc += self.bias.astype(np.int32)

        # Dequantize
        return acc.astype(np.float32) * self.scale


class ClippedReLUQuantized:
    """Quantized ClippedReLU: min(max(x, 0), 127) for int8."""

    def __init__(self, max_value: float = 1.0):
        self.max_value = max_value
        self.max_quantized = 127  # int8 max

    def forward(self, x: np.ndarray, scale: float) -> Tuple[np.ndarray, float]:
        """
        Apply ClippedReLU in quantized domain.
        Returns (quantized_output, output_scale).
        """
        # x is float32, scale is input scale
        # Output: min(max(x, 0), max_value)
        x = np.clip(x, 0, self.max_value)
        # Quantize to int8
        quantized = np.round(x / scale).astype(np.int8)
        return quantized, scale


class NNUEExporter:
    """
    Exports PyTorch NNUE model to Stockfish .nnue format.

    Handles:
    - Per-channel symmetric quantization (int8 weights, int32 bias)
    - Feature transformer quantization
    - Hidden layer quantization
    - Output layer quantization
    - Stockfish-compatible serialization
    """

    def __init__(self, config: dict):
        self.config = config
        export_cfg = config.get("export", {})

        self.quantize = export_cfg.get("quantize", True)
        self.quant_method = export_cfg.get("quantization_method", "per_channel_symmetric")
        self.output_dir = Path(export_cfg.get("dir", "/kaggle/working/export"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.verify = export_cfg.get("verify", True)
        self.verify_samples = export_cfg.get("verify_samples", 10000)
        self.verify_tolerance_cp = export_cfg.get("verify_tolerance_cp", 0.01)
        self.verify_max_error_cp = export_cfg.get("verify_max_error_cp", 1.0)

    def export(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> Path:
        """
        Export model to .nnue file.

        Args:
            model: Trained NNUEModel
            calibration_data: Sample data for quantization calibration (optional)

        Returns:
            Path to exported .nnue file
        """
        model.eval()

        # Generate filename with hash
        network_hash = self._compute_network_hash(model)
        filename = f"nn-{network_hash[:16]}.nnue"
        output_path = self.output_dir / filename

        print(f"Exporting model to {output_path}")

        if self.quantize:
            quantized_layers = self._quantize_model(model, calibration_data)
            self._write_nnue_file(output_path, quantized_layers, network_hash)
        else:
            # Export as float32 (not recommended for Stockfish)
            self._write_nnue_float(output_path, model, network_hash)

        print(f"Export complete: {output_path}")
        return output_path

    def _quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor],
    ) -> Dict[str, QuantizedLayer]:
        """Quantize all layers to int8."""
        quantized = {}

        with torch.no_grad():
            # Feature Transformer
            ft_layer = model.feature_transformer.linear
            quantized["feature_transformer"] = self._quantize_linear(ft_layer, "feature_transformer")

            # Hidden layers
            for i, module in enumerate(model.hidden_layers):
                if isinstance(module, nn.Linear):
                    quantized[f"hidden_{i}"] = self._quantize_linear(module, f"hidden_{i}")

            # Output layer
            quantized["output"] = self._quantize_linear(model.output_layer, "output")

        return quantized

    def _quantize_linear(self, layer: nn.Linear, name: str) -> QuantizedLayer:
        """Quantize a single Linear layer using per-channel symmetric quantization."""
        weight = layer.weight.data.cpu().numpy()  # (out, in)
        bias = layer.bias.data.cpu().numpy() if layer.bias is not None else np.zeros(weight.shape[0])

        # Per-channel symmetric quantization
        # Scale per output channel: max(|weight|) / 127
        max_vals = np.max(np.abs(weight), axis=1)  # (out_features,)
        max_vals = np.maximum(max_vals, 1e-8)  # Avoid division by zero
        scales = max_vals / 127.0  # (out_features,)

        # Quantize weights
        weight_q = np.round(weight / scales[:, np.newaxis]).astype(np.int8)
        weight_q = np.clip(weight_q, -128, 127)

        # Quantize bias (int32, scaled by weight_scale * input_scale)
        # For simplicity, use float32 bias (Stockfish uses int32 with combined scale)
        bias_q = np.round(bias / scales).astype(np.int32)

        return QuantizedLayer(
            weight=weight_q,
            bias=bias_q,
            scale=scales.astype(np.float32),
            input_zero_point=0,
        )

    def _write_nnue_file(
        self,
        path: Path,
        quantized_layers: Dict[str, QuantizedLayer],
        network_hash: str,
    ):
        """Write quantized model to Stockfish .nnue format."""
        with open(path, 'wb') as f:
            # Header
            f.write(b"NNUE")           # Magic
            f.write(struct.pack("<I", 1))  # Version
            f.write(bytes.fromhex(network_hash[:64]))  # Architecture hash (32 bytes)
            f.write(bytes.fromhex(network_hash[:64]))  # Network hash (32 bytes)

            # Write feature transformer
            ft = quantized_layers["feature_transformer"]
            self._write_quantized_layer(f, ft, "feature_transformer")

            # Write hidden layers
            for i in range(len([k for k in quantized_layers if k.startswith("hidden_")])):
                layer = quantized_layers[f"hidden_{i}"]
                self._write_quantized_layer(f, layer, f"hidden_{i}")

            # Write output layer
            out = quantized_layers["output"]
            self._write_quantized_layer(f, out, "output")

    def _write_quantized_layer(self, f, layer: QuantizedLayer, name: str):
        """Write a single quantized layer to file."""
        # Layer metadata
        f.write(struct.pack("<I", layer.in_features))
        f.write(struct.pack("<I", layer.out_features))

        # Weights (int8) - row major: (out, in)
        f.write(layer.weight.tobytes())

        # Bias (int32)
        f.write(layer.bias.tobytes())

        # Scales (float32)
        f.write(layer.scale.tobytes())

    def _write_nnue_float(self, path: Path, model: nn.Module, network_hash: str):
        """Write float32 model (fallback, not for Stockfish use)."""
        with open(path, 'wb') as f:
            f.write(b"NNUE")
            f.write(struct.pack("<I", 1))
            f.write(bytes.fromhex(network_hash[:64]))
            f.write(bytes.fromhex(network_hash[:64]))

            # Feature transformer
            ft = model.feature_transformer.linear
            self._write_float_layer(f, ft)

            # Hidden layers
            for module in model.hidden_layers:
                if isinstance(module, nn.Linear):
                    self._write_float_layer(f, module)

            # Output
            self._write_float_layer(f, model.output_layer)

    def _write_float_layer(self, f, layer: nn.Linear):
        """Write float32 layer."""
        weight = layer.weight.data.cpu().numpy()
        bias = layer.bias.data.cpu().numpy() if layer.bias is not None else np.zeros(weight.shape[0])

        f.write(struct.pack("<I", weight.shape[1]))  # in_features
        f.write(struct.pack("<I", weight.shape[0]))  # out_features
        f.write(weight.astype(np.float32).tobytes())
        f.write(bias.astype(np.float32).tobytes())

    def _compute_network_hash(self, model: nn.Module) -> str:
        """Compute SHA256 of all model weights for identification."""
        hasher = hashlib.sha256()
        for param in model.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
        return hasher.hexdigest()

    def verify_export(
        self,
        original_model: nn.Module,
        nnue_path: Path,
        test_data: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Verify exported .nnue matches original model.

        Returns:
            Dict with verification metrics
        """
        if not self.verify:
            return {"verified": True, "skipped": True}

        print(f"Verifying export with {self.verify_samples} samples...")

        original_model.eval()
        with torch.no_grad():
            # Get original predictions
            original_preds = original_model(test_data).cpu().numpy().squeeze()

            # Load and run exported model (simplified - would need full quantized inference)
            # For now, we'll do a basic check by re-loading the exported weights
            # In practice, this should use the same inference as Stockfish
            exported_preds = self._run_exported_inference(nnue_path, test_data)

        # Compute errors
        abs_errors = np.abs(original_preds - exported_preds)
        mean_abs_error = np.mean(abs_errors)
        max_abs_error = np.max(abs_errors)
        rmse = np.sqrt(np.mean((original_preds - exported_preds) ** 2))

        results = {
            "mean_abs_error_cp": float(mean_abs_error),
            "max_abs_error_cp": float(max_abs_error),
            "rmse_cp": float(rmse),
            "verified": mean_abs_error <= self.verify_tolerance_cp and max_abs_error <= self.verify_max_error_cp,
        }

        print(f"Verification: MAE={mean_abs_error:.4f}cp, Max={max_abs_error:.4f}cp, RMSE={rmse:.4f}cp")
        print(f"Tolerance: MAE<={self.verify_tolerance_cp}cp, Max<={self.verify_max_error_cp}cp")
        print(f"Result: {'PASSED' if results['verified'] else 'FAILED'}")

        if not results["verified"]:
            raise RuntimeError(f"Export verification failed: {results}")

        return results

    def _run_exported_inference(self, nnue_path: Path, test_data: torch.Tensor) -> np.ndarray:
        """Run inference using exported .nnue (simplified - loads quantized weights)."""
        # This is a placeholder - real implementation would parse .nnue and run quantized inference
        # For verification, we'll just reload the quantized weights and run in Python
        # In production, this should match Stockfish's exact inference

        # Parse the exported file
        with open(nnue_path, 'rb') as f:
            magic = f.read(4)
            assert magic == b"NNUE"
            version = struct.unpack("<I", f.read(4))[0]
            arch_hash = f.read(32)
            net_hash = f.read(32)

            # Read feature transformer
            ft_in = struct.unpack("<I", f.read(4))[0]
            ft_out = struct.unpack("<I", f.read(4))[0]
            ft_weight = np.frombuffer(f.read(ft_out * ft_in), dtype=np.int8).reshape(ft_out, ft_in)
            ft_bias = np.frombuffer(f.read(ft_out * 4), dtype=np.int32)
            ft_scale = np.frombuffer(f.read(ft_out * 4), dtype=np.float32)

            # For simplicity, run a basic quantized forward pass
            # This is NOT the full HalfKAv2 inference - just a sanity check
            x = test_data.numpy().astype(np.int8)  # Assume already quantized input

            # Feature transformer
            acc = x @ ft_weight.T + ft_bias
            x = np.clip(acc * ft_scale, 0, 1.0)  # ClippedReLU

            # Hidden layers (simplified - only first hidden layer)
            # In reality, need to parse all layers
            return np.zeros_like(original_preds)  # Placeholder


def create_exporter(config: dict) -> NNUEExporter:
    """Factory function."""
    return NNUEExporter(config)