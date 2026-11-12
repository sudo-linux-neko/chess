# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Rust Binpack Loader - PyO3 Implementation

Drop-in replacement for the C++ loader with:
- No C++ compiler required (uses maturin/pyo3)
- Memory safe
- Same performance as C++
- Easy to install via pip install maturin
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import Optional


RUST_LOADER_DIR = Path(__file__).parent / "rust_loader"
RUST_AVAILABLE = False
_binpack_loader = None


def compile_rust_loader() -> bool:
    """Compile the Rust loader using maturin."""
    global RUST_AVAILABLE, _binpack_loader
    
    if RUST_AVAILABLE:
        return True
    
    try:
        # Check if maturin is available
        subprocess.run(["maturin", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Installing maturin...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install maturin: {e}")
            return False
    
    try:
        # Build the Rust extension
        print("Building Rust binpack loader...")
        result = subprocess.run(
            ["maturin", "develop", "--release", "-m", str(RUST_LOADER_DIR / "Cargo.toml")],
            cwd=RUST_LOADER_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        if result.returncode != 0:
            print(f"Rust build failed: {result.stderr}")
            return False
        
        print("Rust binpack loader built successfully")
        
    except subprocess.TimeoutExpired:
        print("Rust build timed out")
        return False
    except Exception as e:
        print(f"Rust build error: {e}")
        return False
    
    # Try to import
    try:
        import binpack_loader
        _binpack_loader = binpack_loader
        RUST_AVAILABLE = True
        return True
    except ImportError as e:
        print(f"Failed to import Rust loader: {e}")
        return False


def get_rust_loader():
    """Get the compiled Rust loader module."""
    global _binpack_loader, RUST_AVAILABLE
    
    if not RUST_AVAILABLE:
        if not compile_rust_loader():
            return None
    
    return _binpack_loader


class RustBinpackDataset:
    """Python wrapper for Rust BinpackDataset."""
    
    def __init__(self, filepath: str):
        loader = get_rust_loader()
        if loader is None:
            raise RuntimeError("Rust loader not available")
        self._dataset = loader.PyBinpackDataset(filepath)
    
    def __len__(self) -> int:
        return self._dataset.__len__()
    
    def __getitem__(self, index: int):
        return self._dataset.__getitem__(index)
    
    def get_batch(self, indices: list) -> tuple:
        """Get a batch of positions as numpy arrays."""
        return self._dataset.get_batch(indices)


class RustQuantizedNNUE:
    """Python wrapper for Rust QuantizedNNUE inference engine."""
    
    def __init__(self, quantized_layers: dict):
        loader = get_rust_loader()
        if loader is None:
            raise RuntimeError("Rust loader not available")
        
        self._engine = loader.QuantizedNNUE(
            ft_weight=quantized_layers["feature_transformer"].weight,
            ft_bias=quantized_layers["feature_transformer"].bias,
            ft_scale=quantized_layers["feature_transformer"].scale,
            h1_weight=quantized_layers["hidden1"].weight,
            h1_bias=quantized_layers["hidden1"].bias,
            h1_scale=quantized_layers["hidden1"].scale,
            h2_weight=quantized_layers["hidden2"].weight,
            h2_bias=quantized_layers["hidden2"].bias,
            h2_scale=quantized_layers["hidden2"].scale,
            out_weight=quantized_layers["output"].weight,
            out_bias=quantized_layers["output"].bias,
            out_scale=quantized_layers["output"].scale,
        )
    
    def forward(self, features):
        """Run quantized inference."""
        return self._engine.forward(features)


def is_rust_available() -> bool:
    """Check if Rust loader is available."""
    return get_rust_loader() is not None


# Fallback to Python implementation if Rust not available
class PythonBinpackDataset:
    """Pure Python fallback (slow, for testing only)."""
    
    def __init__(self, filepath: str, input_dim: int = 81920):
        self.filepath = filepath
        self.input_dim = input_dim
        self._build_index()
    
    def _build_index(self):
        import struct
        with open(self.filepath, 'rb') as f:
            magic = f.read(4)
            if magic != b"BINP":
                raise ValueError("Invalid binpack file")
            version = struct.unpack("<I", f.read(4))[0]
            self.num_positions = struct.unpack("<Q", f.read(8))[0]
            
            # Build index
            self.position_offsets = []
            pos = f.tell()
            for _ in range(self.num_positions):
                self.position_offsets.append(pos)
                num_features = struct.unpack("<H", f.read(2))[0]
                pos += 2 + num_features * 6 + 13
                f.seek(pos)
    
    def __len__(self):
        return self.num_positions
    
    def __getitem__(self, index: int):
        import struct
        import numpy as np
        
        if index >= self.num_positions:
            raise IndexError
        
        with open(self.filepath, 'rb') as f:
            f.seek(self.position_offsets[index])
            num_features = struct.unpack("<H", f.read(2))[0]
            feature_indices = struct.unpack(f"<{num_features}H", f.read(num_features * 2))
            feature_weights = struct.unpack(f"<{num_features}f", f.read(num_features * 4))
            wdl = struct.unpack("<3f", f.read(12))
            eval_cp = struct.unpack("<f", f.read(4))[0]
            game_result = struct.unpack("<b", f.read(1))[0]
            
            # Convert to dense
            features = np.zeros(self.input_dim, dtype=np.float32)
            for idx, w in zip(feature_indices, feature_weights):
                if idx < self.input_dim:
                    features[idx] = w
            
            return (
                features,
                np.array(wdl, dtype=np.float32),
                np.float32(eval_cp),
                np.int64(game_result),
            )


def create_binpack_dataset(filepath: str, prefer_rust: bool = True):
    """Factory function to create the best available dataset."""
    if prefer_rust and is_rust_available():
        return RustBinpackDataset(filepath)
    else:
        print("Using Python fallback for binpack loading (slow)")
        return PythonBinpackDataset(filepath)


def create_streaming_dataset(
    data_dir: str,
    max_files: int = 5,
    shuffle_buffer_size: int = 5000000,
    seed: int = 42,
    prefer_rust: bool = True,
):
    """Factory function for streaming dataset."""
    if prefer_rust and is_rust_available():
        # Rust streaming dataset not yet implemented, fall back to Python
        pass
    # Fall back to Python implementation in data.py
    from src.data import StreamingBinpackDataset
    return StreamingBinpackDataset(data_dir, max_files, shuffle_buffer_size, seed)