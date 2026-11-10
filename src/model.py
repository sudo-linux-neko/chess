# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
HalfKAv2_hm Network Architecture for Stockfish NNUE.

Based on official-stockfish/nnue-pytorch architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ClippedReLU(nn.Module):
    """Clipped ReLU activation: min(max(x, 0), 1)."""

    def __init__(self, max_value: float = 1.0):
        super().__init__()
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0, self.max_value)


class FeatureTransformer(nn.Module):
    """
    Feature Transformer for HalfKAv2.
    Processes sparse input per king bucket, then concatenates.
    """

    def __init__(
        self,
        input_dim: int,
        ft_size: int,
        num_buckets: int = 64,
        clipped_relu_max: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.ft_size = ft_size
        self.num_buckets = num_buckets
        self.features_per_bucket = input_dim // num_buckets

        # One linear layer per king bucket (shared weights in original, but we use separate for simplicity)
        # Actually, Stockfish uses shared weights across buckets - we'll follow that
        self.linear = nn.Linear(self.features_per_bucket, ft_size, bias=True)
        self.activation = ClippedReLU(clipped_relu_max)

        # Initialize weights (Xavier uniform for stability)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sparse tensor of shape (batch_size, input_dim) or dense (batch_size, input_dim)
        Returns:
            Transformed features of shape (batch_size, num_buckets * ft_size)
        """
        batch_size = x.shape[0]

        # Reshape to (batch_size, num_buckets, features_per_bucket)
        x = x.view(batch_size, self.num_buckets, self.features_per_bucket)

        # Apply shared linear layer per bucket
        # (batch_size, num_buckets, features_per_bucket) -> (batch_size, num_buckets, ft_size)
        x = self.linear(x)
        x = self.activation(x)

        # Flatten: (batch_size, num_buckets * ft_size)
        x = x.view(batch_size, -1)
        return x


class NNUEModel(nn.Module):
    """
    Stockfish-style NNUE network with HalfKAv2_hm features.

    Architecture:
    - Input: 81,920 sparse features (HalfKAv2_hm)
    - Feature Transformer: 64 buckets × 256 -> 16,384
    - L1: 16,384 -> 32 (ClippedReLU)
    - L2: 32 -> 32 (ClippedReLU)
    - Output: 32 -> 1 (Linear, centipawns)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        model_cfg = config.get("model", {})
        input_dim = model_cfg.get("input_dim", 81920)
        ft_size = model_cfg.get("ft_size", 256)
        hidden_sizes = model_cfg.get("hidden_sizes", [32, 32])
        output_size = model_cfg.get("output_size", 1)
        clipped_relu_max = model_cfg.get("clipped_relu_max", 1.0)
        num_buckets = 64  # HalfKAv2 fixed

        # Feature Transformer
        self.feature_transformer = FeatureTransformer(
            input_dim=input_dim,
            ft_size=ft_size,
            num_buckets=num_buckets,
            clipped_relu_max=clipped_relu_max,
        )

        # Hidden layers
        ft_output_size = num_buckets * ft_size  # 64 * 256 = 16384
        layers = []
        prev_size = ft_output_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size, bias=True))
            layers.append(ClippedReLU(clipped_relu_max))
            prev_size = hidden_size

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size, bias=True)

        # Initialize output layer with small weights (Stockfish style)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.01)
        nn.init.zeros_(self.output_layer.bias)

        # Initialize hidden layers
        for module in self.hidden_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, 81920) - can be sparse or dense
        Returns:
            Evaluation in centipawns (batch_size, 1)
        """
        # Feature transformer
        x = self.feature_transformer(x)

        # Hidden layers
        x = self.hidden_layers(x)

        # Output
        x = self.output_layer(x)
        return x

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: dict) -> NNUEModel:
    """Factory function to create model from config."""
    return NNUEModel(config)


# For quantization/export compatibility
class QuantizableNNUEModel(NNUEModel):
    """NNUE model with quantization-aware modules."""

    def __init__(self, config: dict):
        super().__init__(config)
        # Replace activations with quantization-friendly versions if needed
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)
        return x


def fuse_model(model: NNUEModel) -> NNUEModel:
    """Fuse modules for quantization (Conv+BN+ReLU, Linear+ReLU)."""
    # For our architecture: Linear + ClippedReLU
    fused_modules = []
    for name, module in model.hidden_layers.named_children():
        if isinstance(module, nn.Linear):
            # Check if next module is ClippedReLU
            next_modules = list(model.hidden_layers.named_children())
            idx = next_modules.index((name, module))
            if idx + 1 < len(next_modules):
                next_name, next_module = next_modules[idx + 1]
                if isinstance(next_module, ClippedReLU):
                    fused_modules.append([name, next_name])

    if fused_modules:
        torch.quantization.fuse_modules(model.hidden_layers, fused_modules, inplace=True)

    return model