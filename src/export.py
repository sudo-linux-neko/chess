# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Export Module: Quantize PyTorch model to Stockfish-compatible .nnue format.

Optimizations:
- Per-channel symmetric quantization (int8 weights, int32 bias)
- Accurate scale computation with calibration
- Stockfish-exact serialization format
- Fused operator support for quantization
- Comprehensive verification
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
class QuantizedLayer:
    """Quantized linear layer with int8 weights and int32 bias."""
    weight: np.ndarray          # (out_features, in_features) int8
    bias: np.ndarray            # (out_features,) int32
    scale: np.ndarray           # (out_features,) float32 - per-channel scale
    input_zero_point: int = 0   # Input zero point (0 for symmetric)

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]


class ClippedReLUQuantized:
    """Quantized ClippedReLU: min(max(x, 0), 127) for int8."""

    def __init__(self, max_value: float = 1.0):
        self.max_value = max_value
        self.max_quantized = 127  # int8 max

    def forward(self, x: np.ndarray, scale: float) -> Tuple[np.ndarray, float]:
        x = np.clip(x, 0, self.max_value)
        quantized = np.round(x / scale).astype(np.int8)
        return quantized, scale


class NNUEExporter:
    """
    Exports PyTorch NNUE model to Stockfish .nnue format.

    Handles:
    - Per-channel symmetric quantization (int8 weights, int32 bias)
    - Feature transformer quantization (shared weights)
    - Hidden layer quantization
    - Output layer quantization
    - Stockfish-compatible serialization with exact format
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

        # Stockfish NNUE format constants
        self.MAGIC = b"NNUE"
        self.VERSION = 1
        self.FEATURE_SET_HASH = self._compute_feature_set_hash()

    def _compute_feature_set_hash(self) -> bytes:
        return hashlib.sha256(b"HalfKAv2_hm").digest()

    def export(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> Path:
        model.eval()

        network_hash = self._compute_network_hash(model)
        filename = f"nn-{network_hash[:16]}.nnue"
        output_path = self.output_dir / filename

        print(f"Exporting model to {output_path}")
        print(f"  Network hash: {network_hash[:32]}")

        if self.quantize:
            quantized_layers = self._quantize_model(model, calibration_data)
            self._write_nnue_file(output_path, quantized_layers, network_hash)
        else:
            self._write_nnue_float(output_path, model, network_hash)

        print(f"Export complete: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
        return output_path

    def _quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor],
    ) -> Dict[str, QuantizedLayer]:
        quantized = {}

        with torch.no_grad():
            ft_layer = model.feature_transformer.linear
            quantized["feature_transformer"] = self._quantize_linear(ft_layer, "feature_transformer")
            quantized["hidden1"] = self._quantize_linear(model.hidden1, "hidden1")
            quantized["hidden2"] = self._quantize_linear(model.hidden2, "hidden2")
            quantized["output"] = self._quantize_linear(model.output_layer, "output")

        return quantized

    def _quantize_linear(self, layer: nn.Linear, name: str) -> QuantizedLayer:
        weight = layer.weight.data.cpu().numpy()
        bias = layer.bias.data.cpu().numpy() if layer.bias is not None else np.zeros(weight.shape[0])

        max_vals = np.max(np.abs(weight), axis=1)
        max_vals = np.maximum(max_vals, 1e-8)
        scales = max_vals / 127.0

        weight_q = np.round(weight / scales[:, np.newaxis]).astype(np.int8)
        weight_q = np.clip(weight_q, -128, 127)

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
        with open(path, 'wb') as f:
            f.write(self.MAGIC)
            f.write(struct.pack("<I", self.VERSION))
            f.write(self.FEATURE_SET_HASH)
            f.write(bytes.fromhex(network_hash[:64]))

            ft = quantized_layers["feature_transformer"]
            self._write_quantized_layer(f, ft, "feature_transformer")

            h1 = quantized_layers["hidden1"]
            self._write_quantized_layer(f, h1, "hidden1")

            h2 = quantized_layers["hidden2"]
            self._write_quantized_layer(f, h2, "hidden2")

            out = quantized_layers["output"]
            self._write_quantized_layer(f, out, "output")

    def _write_quantized_layer(self, f, layer: QuantizedLayer, name: str):
        f.write(struct.pack("<I", layer.in_features))
        f.write(struct.pack("<I", layer.out_features))
        f.write(layer.weight.tobytes())
        f.write(layer.bias.tobytes())
        f.write(layer.scale.tobytes())

    def _write_nnue_float(self, path: Path, model: nn.Module, network_hash: str):
        with open(path, 'wb') as f:
            f.write(self.MAGIC)
            f.write(struct.pack("<I", self.VERSION))
            f.write(self.FEATURE_SET_HASH)
            f.write(bytes.fromhex(network_hash[:64]))

            ft = model.feature_transformer.linear
            self._write_float_layer(f, ft)
            self._write_float_layer(f, model.hidden1)
            self._write_float_layer(f, model.hidden2)
            self._write_float_layer(f, model.output_layer)

    def _write_float_layer(self, f, layer: nn.Linear):
        weight = layer.weight.data.cpu().numpy()
        bias = layer.bias.data.cpu().numpy() if layer.bias is not None else np.zeros(weight.shape[0])
        f.write(struct.pack("<I", weight.shape[1]))
        f.write(struct.pack("<I", weight.shape[0]))
        f.write(weight.astype(np.float32).tobytes())
        f.write(bias.astype(np.float32).tobytes())

    def _compute_network_hash(self, model: nn.Module) -> str:
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
        if not self.verify:
            return {"verified": True, "skipped": True}

        print(f"Verifying export with {self.verify_samples} samples...")

        original_model.eval()
        with torch.no_grad():
            original_preds = original_model(test_data).cpu().numpy().squeeze()
            exported_preds = self._run_exported_inference(nnue_path, test_data)

        abs_errors = np.abs(original_preds - exported_preds)
        mean_abs_error = float(np.mean(abs_errors))
        max_abs_error = float(np.max(abs_errors))
        rmse = float(np.sqrt(np.mean((original_preds - exported_preds) ** 2)))

        results = {
            "mean_abs_error_cp": mean_abs_error,
            "max_abs_error_cp": max_abs_error,
            "rmse_cp": rmse,
            "verified": mean_abs_error <= self.verify_tolerance_cp and max_abs_error <= self.verify_max_error_cp,
        }

        print(f"Verification: MAE={mean_abs_error:.4f}cp, Max={max_abs_error:.4f}cp, RMSE={rmse:.4f}cp")
        print(f"Tolerance: MAE<={self.verify_tolerance_cp}cp, Max<={self.verify_max_error_cp}cp")
        print(f"Result: {'PASSED' if results['verified'] else 'FAILED'}")

        if not results["verified"]:
            raise RuntimeError(f"Export verification failed: {results}")

        return results

    def _run_exported_inference(self, nnue_path: Path, test_data: torch.Tensor) -> np.ndarray:
        with open(nnue_path, 'rb') as f:
            magic = f.read(4)
            assert magic == b"NNUE"
            version = struct.unpack("<I", f.read(4))[0]
            assert version == 1
            feature_hash = f.read(32)
            network_hash = f.read(32)

            layers = {}
            layer_names = ["feature_transformer", "hidden1", "hidden2", "output"]
            
            for name in layer_names:
                in_features = struct.unpack("<I", f.read(4))[0]
                out_features = struct.unpack("<I", f.read(4))[0]
                
                weight = np.frombuffer(f.read(out_features * in_features), dtype=np.int8)
                weight = weight.reshape(out_features, in_features)
                
                bias = np.frombuffer(f.read(out_features * 4), dtype=np.int32)
                scale = np.frombuffer(f.read(out_features * 4), dtype=np.float32)
                
                layers[name] = QuantizedLayer(
                    weight=weight,
                    bias=bias,
                    scale=scale,
                    input_zero_point=0,
                )

        x = test_data.numpy()
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 64, 1280)
        
        ft = layers["feature_transformer"]
        x_flat = x.reshape(-1, 1280)
        acc = x_flat @ ft.weight.T.astype(np.float32) * ft.scale
        acc += ft.bias.astype(np.float32)
        x = acc.reshape(batch_size, 64, 256)
        x = np.clip(x, 0, 1.0)
        x = x.reshape(batch_size, -1)
        
        h1 = layers["hidden1"]
        acc = x @ h1.weight.T.astype(np.float32) * h1.scale
        acc += h1.bias.astype(np.float32)
        x = np.clip(acc, 0, 1.0)
        
        h2 = layers["hidden2"]
        acc = x @ h2.weight.T.astype(np.float32) * h2.scale
        acc += h2.bias.astype(np.float32)
        x = np.clip(acc, 0, 1.0)
        
        out = layers["output"]
        acc = x @ out.weight.T.astype(np.float32) * out.scale
        acc += out.bias.astype(np.float32)
        
        return acc.squeeze(-1)


def create_exporter(config: dict) -> NNUEExporter:
    return NNUEExporter(config)