# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Export Verification Module.

Verifies that exported .nnue matches PyTorch model within tolerance.
Supports both Python fallback and Rust-accelerated quantized inference.
"""

import struct
import hashlib
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


class ExportVerifier:
    """
    Verifies exported .nnue against original PyTorch model.
    
    Features:
    - Multiple error metrics (MAE, Max, RMSE, Percentiles)
    - Rust-accelerated quantized inference (10-50x faster)
    - Per-layer error analysis
    - Stockfish compatibility check
    """

    def __init__(self, config: dict):
        self.config = config
        verify_cfg = config.get("export", {})

        self.verify_samples = verify_cfg.get("verify_samples", 20000)
        self.tolerance_cp = verify_cfg.get("verify_tolerance_cp", 0.01)
        self.max_error_cp = verify_cfg.get("verify_max_error_cp", 0.5)

        # Try to import Rust quantized inference
        self._rust_available = False
        try:
            from binpack_loader import QuantizedNNUE as RustQuantizedNNUE
            self.RustQuantizedNNUE = RustQuantizedNNUE
            self._rust_available = True
        except ImportError:
            self.RustQuantizedNNUE = None

    def verify(
        self,
        original_model: nn.Module,
        nnue_path: Path,
        test_data: torch.Tensor,
    ) -> Dict[str, Any]:
        """Main verification entry point."""
        original_model.eval()
        original_model = original_model.to(test_data.device)
        
        with torch.no_grad():
            original_preds = original_model(test_data).cpu().numpy().squeeze()
            
            # Use Rust if available for speed
            if self._rust_available:
                exported_preds = self._run_rust_verification(nnue_path, test_data)
            else:
                exported_preds = self._run_python_verification(nnue_path, test_data)

        return self._compute_metrics(original_preds, exported_preds)

    def _compute_metrics(self, original: np.ndarray, exported: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive error metrics."""
        abs_errors = np.abs(original - exported)
        rel_errors = abs_errors / (np.abs(original) + 1e-8)
        
        # Correlation (handle constant arrays gracefully)
        try:
            corr = np.corrcoef(original, exported)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0
        
        metrics = {
            "mean_abs_error_cp": float(np.mean(abs_errors)),
            "max_abs_error_cp": float(np.max(abs_errors)),
            "rmse_cp": float(np.sqrt(np.mean((original - exported) ** 2))),
            "mean_rel_error": float(np.mean(rel_errors)),
            "max_rel_error": float(np.max(rel_errors)),
            "p50_cp": float(np.percentile(abs_errors, 50)),
            "p90_cp": float(np.percentile(abs_errors, 90)),
            "p95_cp": float(np.percentile(abs_errors, 95)),
            "p99_cp": float(np.percentile(abs_errors, 99)),
            "p99.9_cp": float(np.percentile(abs_errors, 99.9)),
            "correlation": float(corr),
            "verified": True,
        }
        
        # Check tolerances
        if metrics["mean_abs_error_cp"] > self.tolerance_cp:
            metrics["verified"] = False
            metrics["failure_reason"] = f"MAE {metrics['mean_abs_error_cp']:.6f} > tolerance {self.tolerance_cp}"
        elif metrics["max_abs_error_cp"] > self.max_error_cp:
            metrics["verified"] = False
            metrics["failure_reason"] = f"Max error {metrics['max_abs_error_cp']:.6f} > tolerance {self.max_error_cp}"
        
        return metrics

    def _run_rust_verification(self, nnue_path: Path, test_data: torch.Tensor) -> np.ndarray:
        """Use Rust SIMD-accelerated quantized inference."""
        # Load quantized parameters from .nnue
        layers = self._load_nnue_layers(nnue_path)
        
        # Prepare data for Rust
        test_np = test_data.cpu().numpy().astype(np.float32)
        
        # Create Rust quantized NNUE - try both possible APIs
        try:
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
        except TypeError:
            # Try alternative API with QuantizedLayer objects
            from src.export import QuantizedLayer
            qlayers = {
                name: QuantizedLayer(
                    weight=layers[name]["weight"],
                    bias=layers[name]["bias"],
                    scale=layers[name]["scale"],
                ) for name in layers
            }
            rust_nnue = self.RustQuantizedNNUE(qlayers)
        
        # Forward pass
        result = rust_nnue.forward(test_np)
        return np.array(result)

    def _run_python_verification(self, nnue_path: Path, test_data: torch.Tensor) -> np.ndarray:
        """Pure Python quantized inference (fallback)."""
        layers = self._load_nnue_layers(nnue_path)
        
        # Get architecture dimensions from loaded layers
        ft = layers["feature_transformer"]
        features_per_bucket = ft["weight"].shape[1]   # 1280 (in_features per bucket)
        ft_output = ft["weight"].shape[0]             # 256 (out_features per bucket)
        h1_in = layers["hidden1"]["weight"].shape[1]  # 16384 (total accumulator)
        num_buckets = h1_in // ft_output               # 64 buckets
        
        x = test_data.cpu().numpy()
        batch_size = x.shape[0]
        x = x.reshape(batch_size, num_buckets, features_per_bucket)
        
        # Feature transformer
        x_flat = x.reshape(-1, features_per_bucket)
        acc = x_flat @ ft["weight"].T.astype(np.float32) * ft["scale"]
        acc += ft["bias"].astype(np.float32)
        x = acc.reshape(batch_size, num_buckets * ft_output)
        x = np.clip(x, 0, 1.0)
        
        # Hidden1
        h1 = layers["hidden1"]
        acc = x @ h1["weight"].T.astype(np.float32) * h1["scale"]
        acc += h1["bias"].astype(np.float32)
        x = np.clip(acc, 0, 1.0)
        
        # Hidden2
        h2 = layers["hidden2"]
        acc = x @ h2["weight"].T.astype(np.float32) * h2["scale"]
        acc += h2["bias"].astype(np.float32)
        x = np.clip(acc, 0, 1.0)
        
        # Output
        out = layers["output"]
        acc = x @ out["weight"].T.astype(np.float32) * out["scale"]
        acc += out["bias"].astype(np.float32)
        
        return acc.squeeze(-1)

    def _load_nnue_layers(self, nnue_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
        """Load quantized layers from .nnue file."""
        file_size = nnue_path.stat().st_size
        with open(nnue_path, 'rb') as f:
            magic = f.read(4)
            if len(magic) < 4:
                raise ValueError(f"Invalid NNUE file: too short (magic)")
            if magic != b"NNUE":
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

        return layers

    def verify_nnue_format(self, nnue_path: Path) -> Dict[str, Any]:
        """Verify .nnue file format compliance with Stockfish."""
        results = {
            "valid_format": True,
            "errors": [],
            "warnings": [],
        }
        
        try:
            with open(nnue_path, 'rb') as f:
                # Check magic
                magic = f.read(4)
                if magic != b"NNUE":
                    results["valid_format"] = False
                    results["errors"].append(f"Invalid magic: {magic}")
                    return results
                
                # Check version
                version = struct.unpack("<I", f.read(4))[0]
                if version != 1:
                    results["warnings"].append(f"Non-standard version: {version}")
                
                # Check feature set hash
                feature_hash = f.read(32)
                expected_hash = hashlib.sha256(b"HalfKAv2_hm").digest()
                if feature_hash != expected_hash:
                    results["warnings"].append(f"Feature set hash mismatch")
                
                network_hash = f.read(32)
                
                # Check layers - validate against expected dims from config
                ft_cfg = self.config.get("model", {})
                num_buckets = ft_cfg.get("num_buckets", 64)
                features_per_bucket = ft_cfg.get("features_per_bucket", 1280)
                ft_output = ft_cfg.get("ft_output", 256)
                hidden1 = ft_cfg.get("hidden1", 256)
                hidden2 = ft_cfg.get("hidden2", 32)
                output_dim = ft_cfg.get("output_dim", 1)
                
                layer_names = ["feature_transformer", "hidden1", "hidden2", "output"]
                expected_dims = [
                    (features_per_bucket, ft_output),   # FT: features_per_bucket -> ft_output
                    (num_buckets * ft_output, hidden1), # H1: accumulator -> hidden1
                    (hidden1, hidden2),                 # H2: hidden1 -> hidden2
                    (hidden2, output_dim),              # Out: hidden2 -> output_dim
                ]
                
                for i, (name, (exp_in, exp_out)) in enumerate(zip(layer_names, expected_dims)):
                    in_features = struct.unpack("<I", f.read(4))[0]
                    out_features = struct.unpack("<I", f.read(4))[0]
                    
                    if in_features != exp_in or out_features != exp_out:
                        results["warnings"].append(
                            f"Layer {name} dims: expected ({exp_in}, {exp_out}), got ({in_features}, {out_features})"
                        )
                    
                    # Skip data
                    f.seek(out_features * in_features + out_features * 4 + out_features * 4, 1)
                
                # Check EOF
                remaining = f.read()
                if remaining:
                    results["warnings"].append(f"Extra data at EOF: {len(remaining)} bytes")
        
        except Exception as e:
            results["valid_format"] = False
            results["errors"].append(f"Format verification failed: {e}")
        
        return results

    def benchmark_inference(
        self,
        nnue_path: Path,
        batch_sizes: List[int] = [1, 8, 32, 128, 512],
        num_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Benchmark quantized inference speed."""
        import time
        
        layers = self._load_nnue_layers(nnue_path)
        
        # Compute input dimension from loaded layers
        ft_in = layers["feature_transformer"]["weight"].shape[1]  # per-bucket features (1280)
        hidden1_in = layers["hidden1"]["weight"].shape[1]  # total accumulator (16384)
        num_buckets = hidden1_in // layers["feature_transformer"]["weight"].shape[0]  # 64
        input_dim = ft_in * num_buckets  # 81920
        
        results = {}
        for batch_size in batch_sizes:
            # Create random test data
            test_data = np.random.randn(batch_size, input_dim).astype(np.float32)
            
            # Warmup
            for _ in range(5):
                _ = self._run_python_verification_single(layers, test_data)
            
            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = self._run_python_verification_single(layers, test_data)
                times.append(time.perf_counter() - start)
            
            times = np.array(times)
            positions_per_sec = batch_size / np.mean(times)
            
            results[batch_size] = {
                "mean_ms": float(np.mean(times) * 1000),
                "std_ms": float(np.std(times) * 1000),
                "min_ms": float(np.min(times) * 1000),
                "max_ms": float(np.max(times) * 1000),
                "positions_per_sec": float(positions_per_sec),
            }
        
        return results

    def _run_python_verification_single(self, layers: Dict, test_data: np.ndarray) -> np.ndarray:
        """Single forward pass for benchmarking."""
        x = test_data
        batch_size = x.shape[0]
        
        # Get architecture dimensions from loaded layers
        ft = layers["feature_transformer"]
        features_per_bucket = ft["weight"].shape[1]   # 1280
        ft_output = ft["weight"].shape[0]             # 256
        h1_in = layers["hidden1"]["weight"].shape[1]  # 16384
        num_buckets = h1_in // ft_output                # 64
        
        x = x.reshape(batch_size, num_buckets, features_per_bucket)
        
        # Feature transformer
        x_flat = x.reshape(-1, features_per_bucket)
        acc = x_flat @ ft["weight"].T.astype(np.float32) * ft["scale"]
        acc += ft["bias"].astype(np.float32)
        x = acc.reshape(batch_size, num_buckets * ft_output)
        x = np.clip(x, 0, 1.0)
        
        # Hidden1
        h1 = layers["hidden1"]
        acc = x @ h1["weight"].T.astype(np.float32) * h1["scale"]
        acc += h1["bias"].astype(np.float32)
        x = np.clip(acc, 0, 1.0)
        
        # Hidden2
        h2 = layers["hidden2"]
        acc = x @ h2["weight"].T.astype(np.float32) * h2["scale"]
        acc += h2["bias"].astype(np.float32)
        x = np.clip(acc, 0, 1.0)
        
        # Output
        out = layers["output"]
        acc = x @ out["weight"].T.astype(np.float32) * out["scale"]
        acc += out["bias"].astype(np.float32)
        
        return acc.squeeze(-1)


def verify_export_standalone(
    config: dict,
    model_path: Path,
    nnue_path: Path,
    test_data: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Standalone verification function for CLI."""
    from src.model import create_model
    from src.checkpoint import CheckpointManager
    
    # Load model
    model = create_model(config)
    checkpoint_manager = CheckpointManager(config)
    checkpoint_manager.load_checkpoint(model_path, model=model)
    
    # Generate test data if not provided
    if test_data is None:
        test_data = torch.randn(config.get("export", {}).get("verify_samples", 20000), 
                               config.get("model", {}).get("input_dim", 81920))
    
    # Verify
    verifier = ExportVerifier(config)
    results = verifier.verify(model, nnue_path, test_data)
    
    # Also verify format
    format_results = verifier.verify_nnue_format(nnue_path)
    results["format_check"] = format_results
    
    return results