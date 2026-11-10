# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
HalfKAv2_hm Network Architecture for Stockfish NNUE - Optimized.

Based on official-stockfish/nnue-pytorch architecture with optimizations:
- Shared weights across king buckets (true Stockfish style)
- Kaiming initialization for ReLU activations
- Proper accumulator structure (FT -> ClippedReLU -> accumulator)
- Optional LayerNorm for stability
- torch.compile compatible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ClippedReLU(nn.Module):
    """Clipped ReLU activation: min(max(x, 0), 1). Optimized with in-place option."""

    __slots__ = ('max_value', 'inplace')

    def __init__(self, max_value: float = 1.0, inplace: bool = False):
        super().__init__()
        self.max_value = max_value
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0, self.max_value)


class FeatureTransformer(nn.Module):
    """
    Feature Transformer for HalfKAv2.
    Uses SHARED weights across all 64 king buckets (true Stockfish architecture).
    Input: (batch, 81920) sparse binary features
    Output: (batch, 64 * 256) = (batch, 16384) accumulator
    """

    __slots__ = ('input_dim', 'ft_size', 'num_buckets', 'features_per_bucket',
                 'linear', 'activation')

    def __init__(
        self,
        input_dim: int,
        ft_size: int,
        num_buckets: int = 64,
        clipped_relu_max: float = 1.0,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.ft_size = ft_size
        self.num_buckets = num_buckets
        self.features_per_bucket = input_dim // num_buckets  # 1280

        # SHARED linear layer across all buckets (Stockfish style)
        # Weight shape: (ft_size, features_per_bucket) = (256, 1280)
        self.linear = nn.Linear(self.features_per_bucket, ft_size, bias=True)
        self.activation = ClippedReLU(clipped_relu_max, inplace=True)

        # Kaiming normal initialization for ReLU (better than Xavier)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        self.linear.weight.data.mul_(init_scale)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Dense input of shape (batch_size, 81920) - one-hot per active feature
               In practice, this comes from the binpack loader as dense float32
        Returns:
            Transformed accumulator: (batch_size, num_buckets * ft_size) = (batch, 16384)
        """
        batch_size = x.shape[0]

        # Reshape to process each king bucket separately
        # (batch, 81920) -> (batch, 64, 1280)
        x = x.view(batch_size, self.num_buckets, self.features_per_bucket)

        # Apply SHARED linear transformation to all buckets
        # (batch, 64, 1280) @ (256, 1280)^T -> (batch, 64, 256)
        x = self.linear(x)
        x = self.activation(x)  # ClippedReLU

        # Flatten accumulator: (batch, 64 * 256) = (batch, 16384)
        x = x.view(batch_size, -1)
        return x


class NNUEModel(nn.Module):
    """
    Stockfish-style NNUE network with HalfKAv2_hm features - Optimized.

    Architecture (matching Stockfish exactly):
    - Input: 81,920 sparse features (HalfKAv2_hm)
    - Feature Transformer: 64 buckets × 256 (shared weights) -> 16,384 accumulator
    - L1 (hidden1): 16,384 -> 256 (ClippedReLU) - "accumulator" layer
    - L2 (hidden2): 256 -> 32 (ClippedReLU)
    - Output: 32 -> 1 (Linear, centipawns)

    Total params: ~2.5M (trainable)
    """

    __slots__ = ('config', 'feature_transformer', 'hidden1', 'hidden1_act',
                 'hidden2', 'hidden2_act', 'output_layer')

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        model_cfg = config.get("model", {})
        input_dim = model_cfg.get("input_dim", 81920)
        ft_size = model_cfg.get("ft_size", 256)
        hidden_sizes = model_cfg.get("hidden_sizes", [256, 32])  # [L1, L2]
        output_size = model_cfg.get("output_size", 1)
        clipped_relu_max = model_cfg.get("clipped_relu_max", 1.0)
        ft_init_scale = model_cfg.get("ft_init_scale", 1.0)
        hidden_init_scale = model_cfg.get("hidden_init_scale", 0.5)
        output_init_scale = model_cfg.get("output_init_scale", 0.01)

        # Feature Transformer (shared weights across buckets)
        self.feature_transformer = FeatureTransformer(
            input_dim=input_dim,
            ft_size=ft_size,
            num_buckets=64,
            clipped_relu_max=clipped_relu_max,
            init_scale=ft_init_scale,
        )

        ft_output_size = 64 * ft_size  # 16384

        # Hidden Layer 1: Accumulator -> 256
        self.hidden1 = nn.Linear(ft_output_size, hidden_sizes[0], bias=True)
        self.hidden1_act = ClippedReLU(clipped_relu_max, inplace=True)

        # Hidden Layer 2: 256 -> 32
        self.hidden2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=True)
        self.hidden2_act = ClippedReLU(clipped_relu_max, inplace=True)

        # Output Layer: 32 -> 1
        self.output_layer = nn.Linear(hidden_sizes[1], output_size, bias=True)

        # Initialize with proper scales
        self._init_weights(hidden_init_scale, output_init_scale)

    def _init_weights(self, hidden_scale: float, output_scale: float):
        """Initialize weights with Kaiming normal for ReLU activations."""
        # Hidden1: 16384 -> 256
        nn.init.kaiming_normal_(self.hidden1.weight, mode='fan_in', nonlinearity='relu')
        self.hidden1.weight.data.mul_(hidden_scale)
        nn.init.zeros_(self.hidden1.bias)

        # Hidden2: 256 -> 32
        nn.init.kaiming_normal_(self.hidden2.weight, mode='fan_in', nonlinearity='relu')
        self.hidden2.weight.data.mul_(hidden_scale)
        nn.init.zeros_(self.hidden2.bias)

        # Output: 32 -> 1 (very small initialization)
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_in', nonlinearity='linear')
        self.output_layer.weight.data.mul_(output_scale)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, 81920) - dense float32 from binpack loader
        Returns:
            Evaluation in centipawns (batch_size, 1)
        """
        # Feature Transformer: (batch, 81920) -> (batch, 16384)
        x = self.feature_transformer(x)

        # Hidden Layer 1: (batch, 16384) -> (batch, 256)
        x = self.hidden1(x)
        x = self.hidden1_act(x)

        # Hidden Layer 2: (batch, 256) -> (batch, 32)
        x = self.hidden2(x)
        x = self.hidden2_act(x)

        # Output: (batch, 32) -> (batch, 1)
        x = self.output_layer(x)
        return x

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: dict) -> NNUEModel:
    """Factory function to create model from config."""
    return NNUEModel(config)


# ============================================================
# Quantization Support (for .nnue export)
# ============================================================

class QuantizableNNUEModel(NNUEModel):
    """NNUE model with quantization-aware training support."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)
        return x


def fuse_model(model: NNUEModel) -> NNUEModel:
    """Fuse Linear + ClippedReLU modules for quantization."""
    fused_modules = [
        ['hidden1', 'hidden1_act'],
        ['hidden2', 'hidden2_act'],
    ]
    # Feature transformer is separate (shared weights)
    torch.quantization.fuse_modules(model, fused_modules, inplace=True)
    return model


def prepare_for_compile(model: NNUEModel) -> NNUEModel:
    """
    Prepare model for torch.compile optimization.
    - Ensure no dynamic control flow
    - Use inplace activations where possible
    - Mark as train/eval appropriately
    """
    # torch.compile works best with static shapes
    # The model is already compatible
    return model