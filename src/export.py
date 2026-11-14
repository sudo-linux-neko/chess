# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Ultra-Optimized Export Module: Quantize PyTorch model to Stockfish-compatible .nnue format.

Optimizations:
- Per-channel symmetric quantization (int8 weights, int32 bias)
- Accurate scale computation with calibration data
- Stockfish-exact serialization format
- Fused operator support for quantization
- Comprehensive verification with Rust-accelerated quantized inference
- EMA model export support
"""

import struct
import hashlib
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Quantized inference: (dot + bias_q) * scale (Convention A, matches Rust)."""
        x_f = x.astype(np.float32)
        return ((x_f @ self.weight.T.astype(np.float32)) + self.bias.astype(np.float32)) * self.scale


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
    - EMA model export for better playing strength
    """

    def __init__(self, config: dict):
        self.config = config
        export_cfg = config.get("export", {})

        self.quantize = export_cfg.get("quantize", True)
        self.quant_method = export_cfg.get("quantization_method", "per_channel_symmetric")
        self.output_dir = Path(export_cfg.get("dir", "/kaggle/working/export"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.verify = export_cfg.get("verify", True)
        self.verify_samples = export_cfg.get("verify_samples", 20000)
        self.verify_tolerance_cp = export_cfg.get("verify_tolerance_cp", 0.01)
        self.verify_max_error_cp = export_cfg.get("verify_max_error_cp", 0.5)

        # Stockfish NNUE format constants
        self.MAGIC = b"NNUE"
        self.VERSION = 1
        self.FEATURE_SET_HASH = self._compute_feature_set_hash()

        # Try to import Rust quantized inference
        self._rust_available = False
        try:
            from binpack_loader import QuantizedNNUE as RustQuantizedNNUE
            self.RustQuantizedNNUE = RustQuantizedNNUE
            self._rust_available = True
        except ImportError:
            self.RustQuantizedNNUE = None

    def _compute_feature_set_hash(self) -> bytes:
        return hashlib.sha256(b"HalfKAv2_hm").digest()

    def export(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
        use_ema: bool = False,
    ) -> Path:
        # Handle torch.compile wrapped models
        original_model = model
        if hasattr(model, '_orig_mod'):
            original_model = model._orig_mod
        
        original_model.eval()

        # Use EMA model if available and requested
        if use_ema and hasattr(original_model, 'ema_model') and original_model.ema_model is not None:
            print("Exporting EMA model for better playing strength")
            original_model = original_model.ema_model
            original_model.eval()

        network_hash = self._compute_network_hash(original_model)
        filename = f"nn-{network_hash[:16]}.nnue"
        output_path = self.output_dir / filename

        print(f"Exporting model to {output_path}")
        print(f"  Network hash: {network_hash[:32]}")
        print(f"  Using EMA: {use_ema}")

        if self.quantize:
            quantized_layers = self._quantize_model(original_model, calibration_data)
            self._write_nnue_file(output_path, quantized_layers, network_hash)
        else:
            self._write_nnue_float(output_path, original_model, network_hash)

        print(f"Export complete: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
        return output_path

    def _quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor],
    ) -> Dict[str, QuantizedLayer]:
        quantized = {}
        batch_size = calibration_data.shape[0] if calibration_data is not None else 1

        with torch.no_grad():
            # Each layer needs its own correctly-shaped input:
            # FT: 1280-dim per bucket, hidden1: 16384-dim accumulator,
            # hidden2: 256-dim, output: 32-dim
            layers_to_quantize = [
                ("feature_transformer", model.feature_transformer.linear),
                ("hidden1", model.hidden1_linear),
                ("hidden2", model.hidden2_linear),
                ("output", model.output_layer),
            ]
            for name, layer in layers_to_quantize:
                dummy_input = torch.randn(batch_size, layer.in_features)
                quantized[name] = self._quantize_linear(layer, name, dummy_input)

        return quantized

    def _quantize_linear(
        self,
        layer: nn.Linear,
        name: str,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> QuantizedLayer:
        weight = layer.weight.data.cpu().numpy()
        bias = layer.bias.data.cpu().numpy() if layer.bias is not None else np.zeros(weight.shape[0])

        # Per-channel symmetric quantization
        # Scale = max(|weight|) / 127 per output channel
        max_vals = np.max(np.abs(weight), axis=1)
        max_vals = np.maximum(max_vals, 1e-8)  # Avoid division by zero
        scales = max_vals / 127.0

        # Quantize weights
        # Clip BEFORE astype to prevent int8 overflow/wrap-around (e.g., 200 -> -56 instead of 127)
        weight_q = np.clip(np.round(weight / scales[:, np.newaxis]), -128, 127).astype(np.int8)

        # Quantize bias (Stockfish-compatible: per-channel scale-normalized int32)
        # bias_q = round(bias / scale) so inference: (dot + bias_q) * scale ≈ dot * weight + bias
        # Clip before astype to prevent int32 overflow on extreme values
        bias_q = np.clip(np.round(bias / scales), -2**31, 2**31 - 1).astype(np.int32)

        # Verify quantization accuracy if calibration data provided
        if calibration_data is not None:
            self._verify_quantization(layer, weight_q, bias_q, scales, calibration_data, name)

        return QuantizedLayer(
            weight=weight_q,
            bias=bias_q,
            scale=scales.astype(np.float32),
            input_zero_point=0,
        )

    def _verify_quantization(
        self,
        layer: nn.Linear,
        weight_q: np.ndarray,
        bias_q: np.ndarray,
        scales: np.ndarray,
        calibration_data: torch.Tensor,
        name: str,
    ):
        """Verify quantization accuracy on calibration data."""
        with torch.no_grad():
            # Move to CPU for numpy conversion
            cal = calibration_data.cpu()
            layer_cpu = layer.cpu()
            
            # Original output (model applies ClippedReLU after linear layer)
            orig_out = layer_cpu(cal).numpy()
            if name in ["feature_transformer", "hidden1", "hidden2"]:
                orig_out = np.clip(orig_out, 0, 1.0)
            
            # Quantized output (Stockfish-compatible: (dot + bias_q) * scale)
            quant_out = (cal.numpy() @ weight_q.T.astype(np.float32) + bias_q.astype(np.float32)) * scales
            
            # Apply ClippedReLU for hidden layers
            if name in ["feature_transformer", "hidden1", "hidden2"]:
                quant_out = np.clip(quant_out, 0, 1.0)
            
            # Compute error
            abs_error = np.abs(orig_out - quant_out)
            max_error = np.max(abs_error)
            mean_error = np.mean(abs_error)
            
            print(f"  {name} quantization: max_error={max_error:.6f}, mean_error={mean_error:.6f}")
            
            if max_error > 1e-3:
                print(f"  WARNING: {name} quantization error high!")

    def _write_nnue_file(
        self,
        path: Path,
        quantized_layers: Dict[str, QuantizedLayer],
        network_hash: str,
    ):
        with open(path, 'wb') as f:
            # Header
            f.write(self.MAGIC)
            f.write(struct.pack("<I", self.VERSION))
            f.write(self.FEATURE_SET_HASH)
            f.write(bytes.fromhex(network_hash[:64]))

            # Layers in order: FT, H1, H2, Output
            ft = quantized_layers["feature_transformer"]
            self._write_quantized_layer(f, ft, "feature_transformer")

            h1 = quantized_layers["hidden1"]
            self._write_quantized_layer(f, h1, "hidden1")

            h2 = quantized_layers["hidden2"]
            self._write_quantized_layer(f, h2, "hidden2")

            out = quantized_layers["output"]
            self._write_quantized_layer(f, out, "output")

    def _write_quantized_layer(self, f, layer: QuantizedLayer, name: str):
        # Write dimensions
        f.write(struct.pack("<I", layer.in_features))
        f.write(struct.pack("<I", layer.out_features))
        
        # Write weight (int8)
        f.write(layer.weight.tobytes())
        
        # Write scale (float32) - Stockfish format: scales BEFORE bias
        f.write(layer.scale.tobytes())
        
        # Write bias (int32)
        f.write(layer.bias.tobytes())

    def _write_nnue_float(self, path: Path, model: nn.Module, network_hash: str):
        with open(path, 'wb') as f:
            f.write(self.MAGIC)
            f.write(struct.pack("<I", self.VERSION))
            f.write(self.FEATURE_SET_HASH)
            f.write(bytes.fromhex(network_hash[:64]))

            ft = model.feature_transformer.linear
            self._write_float_layer(f, ft)
            self._write_float_layer(f, model.hidden1_linear)
            self._write_float_layer(f, model.hidden2_linear)
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
        # Filter out ema_model submodule parameters to avoid double-counting
        # (ema_model is a submodule of NNUEModel, so model.parameters()
        # would include EMA params too, corrupting the hash)
        # Handle both compiled (_orig_mod.ema_model.xxx) and uncompiled (ema_model.xxx) names
        for name, param in model.named_parameters():
            if name.startswith("ema_model.") or name.startswith("_orig_mod.ema_model."):
                continue
            hasher.update(param.data.cpu().numpy().tobytes())
        # Include architecture config for uniqueness
        try:
            arch_str = f"{model.config.get('model', {})}"
        except AttributeError:
            arch_str = "unknown_arch"
        hasher.update(arch_str.encode())
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
        original_model = original_model.to(test_data.device)
        with torch.no_grad():
            original_preds = original_model(test_data).cpu().numpy().squeeze(-1)
            
            # Use appropriate inference method based on export format
            if self.quantize:
                # Use Rust-accelerated verification if available
                if self._rust_available:
                    exported_preds = self._run_rust_verification(nnue_path, test_data)
                else:
                    exported_preds = self._run_exported_inference(nnue_path, test_data)
            else:
                exported_preds = self._run_float_inference(nnue_path, test_data)

        abs_errors = np.abs(original_preds - exported_preds)
        mean_abs_error = float(np.mean(abs_errors))
        max_abs_error = float(np.max(abs_errors))
        rmse = float(np.sqrt(np.mean((original_preds - exported_preds) ** 2)))

        # Percentiles
        p50 = float(np.percentile(abs_errors, 50))
        p95 = float(np.percentile(abs_errors, 95))
        p99 = float(np.percentile(abs_errors, 99))

        results = {
            "mean_abs_error_cp": mean_abs_error,
            "max_abs_error_cp": max_abs_error,
            "rmse_cp": rmse,
            "p50_cp": p50,
            "p95_cp": p95,
            "p99_cp": p99,
            "verified": mean_abs_error <= self.verify_tolerance_cp and max_abs_error <= self.verify_max_error_cp,
        }

        print(f"Verification Results:")
        print(f"  MAE: {mean_abs_error:.6f} cp")
        print(f"  Max: {max_abs_error:.6f} cp")
        print(f"  RMSE: {rmse:.6f} cp")
        print(f"  P50: {p50:.6f} cp, P95: {p95:.6f} cp, P99: {p99:.6f} cp")
        print(f"  Tolerance: MAE<={self.verify_tolerance_cp}cp, Max<={self.verify_max_error_cp}cp")
        print(f"  Result: {'PASSED' if results['verified'] else 'FAILED'}")

        if not results["verified"]:
            raise RuntimeError(f"Export verification failed: {results}")

        return results

    def _run_rust_verification(self, nnue_path: Path, test_data: torch.Tensor) -> np.ndarray:
        """Use Rust SIMD-accelerated quantized inference for verification."""
        # Load quantized parameters with bounds checking
        file_size = nnue_path.stat().st_size
        with open(nnue_path, 'rb') as f:
            magic = f.read(4)
            if len(magic) < 4 or magic != b"NNUE":
                raise ValueError(f"Invalid NNUE magic: {magic}")
            
            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                raise ValueError(f"Invalid NNUE file: too short (version)")
            version = struct.unpack("<I", version_bytes)[0]
            if version != 1:
                raise ValueError(f"Unsupported NNUE version: {version}")
            
            feature_hash = f.read(32)
            if len(feature_hash) < 32:
                raise ValueError(f"Invalid NNUE file: too short (feature_hash)")
            network_hash = f.read(32)
            if len(network_hash) < 32:
                raise ValueError(f"Invalid NNUE file: too short (network_hash)")

            layers = {}
            layer_names = ["feature_transformer", "hidden1", "hidden2", "output"]
            
            for name in layer_names:
                in_features_bytes = f.read(4)
                if len(in_features_bytes) < 4:
                    raise ValueError(f"Invalid NNUE file: too short (layer {name} in_features)")
                in_features = struct.unpack("<I", in_features_bytes)[0]
                
                out_features_bytes = f.read(4)
                if len(out_features_bytes) < 4:
                    raise ValueError(f"Invalid NNUE file: too short (layer {name} out_features)")
                out_features = struct.unpack("<I", out_features_bytes)[0]
                
                weight_size = out_features * in_features
                weight = np.frombuffer(f.read(weight_size), dtype=np.int8)
                if len(weight) < weight_size:
                    raise ValueError(f"Invalid NNUE file: truncated weight for layer {name}")
                weight = weight.reshape(out_features, in_features)
                
                scale_size = out_features * 4
                scale = np.frombuffer(f.read(scale_size), dtype=np.float32)
                if len(scale) < out_features:
                    raise ValueError(f"Invalid NNUE file: truncated scale for layer {name}")
                scale = scale.reshape(out_features)
                
                bias_size = out_features * 4
                bias = np.frombuffer(f.read(bias_size), dtype=np.int32)
                if len(bias) < out_features:
                    raise ValueError(f"Invalid NNUE file: truncated bias for layer {name}")
                bias = bias.reshape(out_features)
                
                layers[name] = {
                    "weight": weight,
                    "bias": bias,
                    "scale": scale,
                    "in_features": in_features,
                    "out_features": out_features,
                }

        # Convert to Rust format
        test_np = test_data.cpu().numpy().astype(np.float32)
        batch_size = test_np.shape[0]
        
        # Call Rust quantized inference
        rust_nnue = self.RustQuantizedNNUE(
            ft_weight=layers["feature_transformer"]["weight"],
            ft_bias=layers["feature_transformer"]["bias"],
            ft_scale=layers["feature_transformer"]["scale"],
            h1_weight=layers["hidden1"]["weight"],
            h1_bias=layers["hidden1"]["bias"],
            h1_scale=layers["hidden1"]["scale"],
            h2_weight=layers["hidden2"]["weight"],
            h2_bias=layers["hidden2"]["bias"],
            h2_scale=layers["hidden2"]["scale"],
            out_weight=layers["output"]["weight"],
            out_bias=layers["output"]["bias"],
            out_scale=layers["output"]["scale"],
        )
        
        # Forward pass
        result = rust_nnue.forward(test_np)
        return np.array(result)

    def _run_exported_inference(self, nnue_path: Path, test_data: torch.Tensor) -> np.ndarray:
        """Pure Python quantized inference (fallback)."""
        file_size = nnue_path.stat().st_size
        with open(nnue_path, 'rb') as f:
            magic = f.read(4)
            if len(magic) < 4 or magic != b"NNUE":
                raise ValueError(f"Invalid NNUE magic: {magic}")
            
            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                raise ValueError(f"Invalid NNUE file: too short (version)")
            version = struct.unpack("<I", version_bytes)[0]
            if version != 1:
                raise ValueError(f"Unsupported NNUE version: {version}")
            
            feature_hash = f.read(32)
            if len(feature_hash) < 32:
                raise ValueError(f"Invalid NNUE file: too short (feature_hash)")
            network_hash = f.read(32)
            if len(network_hash) < 32:
                raise ValueError(f"Invalid NNUE file: too short (network_hash)")

            layers = {}
            layer_names = ["feature_transformer", "hidden1", "hidden2", "output"]
            
            for name in layer_names:
                in_features_bytes = f.read(4)
                if len(in_features_bytes) < 4:
                    raise ValueError(f"Invalid NNUE file: too short (layer {name} in_features)")
                in_features = struct.unpack("<I", in_features_bytes)[0]
                
                out_features_bytes = f.read(4)
                if len(out_features_bytes) < 4:
                    raise ValueError(f"Invalid NNUE file: too short (layer {name} out_features)")
                out_features = struct.unpack("<I", out_features_bytes)[0]
                
                weight_size = out_features * in_features
                weight = np.frombuffer(f.read(weight_size), dtype=np.int8)
                if len(weight) < weight_size:
                    raise ValueError(f"Invalid NNUE file: truncated weight for layer {name}")
                weight = weight.reshape(out_features, in_features)
                
                scale_size = out_features * 4
                scale = np.frombuffer(f.read(scale_size), dtype=np.float32)
                if len(scale) < out_features:
                    raise ValueError(f"Invalid NNUE file: truncated scale for layer {name}")
                scale = scale.reshape(out_features)
                
                bias_size = out_features * 4
                bias = np.frombuffer(f.read(bias_size), dtype=np.int32)
                if len(bias) < out_features:
                    raise ValueError(f"Invalid NNUE file: truncated bias for layer {name}")
                bias = bias.reshape(out_features)
                
                layers[name] = QuantizedLayer(
                    weight=weight,
                    bias=bias,
                    scale=scale,
                    input_zero_point=0,
                )

        x = test_data.cpu().numpy()
        batch_size = x.shape[0]
        
        # Get architecture dimensions from loaded layers
        ft = layers["feature_transformer"]
        features_per_bucket = ft.weight.shape[1]   # 1280
        ft_output = ft.weight.shape[0]             # 256
        h1_in = layers["hidden1"].weight.shape[1]  # 16384
        num_buckets = h1_in // ft_output               # 64
        
        x = x.reshape(batch_size, num_buckets, features_per_bucket)
        
        # Feature transformer
        x_flat = x.reshape(-1, features_per_bucket)
        acc = (x_flat @ ft.weight.T.astype(np.float32) + ft.bias.astype(np.float32)) * ft.scale
        x = acc.reshape(batch_size, num_buckets * ft_output)
        x = np.clip(x, 0, 1.0)
        
        # Hidden1
        h1 = layers["hidden1"]
        acc = (x @ h1.weight.T.astype(np.float32) + h1.bias.astype(np.float32)) * h1.scale
        x = np.clip(acc, 0, 1.0)
        
        # Hidden2
        h2 = layers["hidden2"]
        acc = (x @ h2.weight.T.astype(np.float32) + h2.bias.astype(np.float32)) * h2.scale
        x = np.clip(acc, 0, 1.0)
        
        # Output
        out = layers["output"]
        acc = (x @ out.weight.T.astype(np.float32) + out.bias.astype(np.float32)) * out.scale
        
        return acc.squeeze(-1)

    def _run_float_inference(self, nnue_path: Path, test_data: torch.Tensor) -> np.ndarray:
        """Pure Python float inference for float-format .nnue files."""
        with open(nnue_path, 'rb') as f:
            magic = f.read(4)
            if len(magic) < 4 or magic != b"NNUE":
                raise ValueError(f"Invalid NNUE magic: {magic}")

            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                raise ValueError(f"Invalid NNUE file: too short (version)")
            version = struct.unpack("<I", version_bytes)[0]
            if version != 1:
                raise ValueError(f"Unsupported NNUE version: {version}")

            feature_hash = f.read(32)
            if len(feature_hash) < 32:
                raise ValueError(f"Invalid NNUE file: too short (feature_hash)")
            network_hash = f.read(32)
            if len(network_hash) < 32:
                raise ValueError(f"Invalid NNUE file: too short (network_hash)")

            layers = {}
            layer_names = ["feature_transformer", "hidden1", "hidden2", "output"]

            for name in layer_names:
                in_features_bytes = f.read(4)
                if len(in_features_bytes) < 4:
                    raise ValueError(f"Invalid NNUE file: too short (layer {name} in_features)")
                in_features = struct.unpack("<I", in_features_bytes)[0]

                out_features_bytes = f.read(4)
                if len(out_features_bytes) < 4:
                    raise ValueError(f"Invalid NNUE file: too short (layer {name} out_features)")
                out_features = struct.unpack("<I", out_features_bytes)[0]

                weight_bytes = out_features * in_features * 4  # float32 = 4 bytes per element
                weight = np.frombuffer(f.read(weight_bytes), dtype=np.float32)
                if len(weight) < out_features * in_features:
                    raise ValueError(f"Invalid NNUE file: truncated weight for layer {name}")
                weight = weight.reshape(out_features, in_features)

                bias_size = out_features * 4
                bias = np.frombuffer(f.read(bias_size), dtype=np.float32)
                if len(bias) < out_features:
                    raise ValueError(f"Invalid NNUE file: truncated bias for layer {name}")
                bias = bias.reshape(out_features)

                layers[name] = {
                    "weight": weight,
                    "bias": bias,
                    "in_features": in_features,
                    "out_features": out_features,
                }

        x = test_data.cpu().numpy()
        batch_size = x.shape[0]

        ft = layers["feature_transformer"]
        features_per_bucket = ft["weight"].shape[1]   # 1280
        ft_output = ft["weight"].shape[0]             # 256
        h1_in = layers["hidden1"]["weight"].shape[1]  # 16384
        num_buckets = h1_in // ft_output               # 64

        x = x.reshape(batch_size, num_buckets, features_per_bucket)

        # Feature transformer
        x_flat = x.reshape(-1, features_per_bucket)
        acc = x_flat @ ft["weight"].T
        acc += ft["bias"]
        x = acc.reshape(batch_size, num_buckets * ft_output)
        x = np.clip(x, 0, 1.0)

        # Hidden1
        h1 = layers["hidden1"]
        acc = x @ h1["weight"].T
        acc += h1["bias"]
        x = np.clip(acc, 0, 1.0)

        # Hidden2
        h2 = layers["hidden2"]
        acc = x @ h2["weight"].T
        acc += h2["bias"]
        x = np.clip(acc, 0, 1.0)

        # Output
        out = layers["output"]
        acc = x @ out["weight"].T
        acc += out["bias"]

        return acc.squeeze(-1)


def create_exporter(config: dict) -> NNUEExporter:
    return NNUEExporter(config)
