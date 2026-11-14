# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Ultra-Optimized NNUE Model Architecture for Maximum Elo.

Based on official Stockfish nnue-pytorch and nettest best practices:
- HalfKAv2_hm feature set
- Kaiming Normal initialization (critical for ClippedReLU)
- Proper ClippedReLU implementation
- torch.compile compatibility
- EMA support for evaluation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import math


def _get_clipped_relu():
    """Get optimized ClippedReLU - uses torch.compile compatible implementation."""
    return torch.clamp


class ClippedReLU(nn.Module):
    """ClippedReLU: min(max(x, 0), 1) - Stockfish standard activation."""

    def __init__(self, max_value: float = 1.0):
        super().__init__()
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=0.0, max=self.max_value)

    def extra_repr(self) -> str:
        return f"max_value={self.max_value}"


class FeatureTransformer(nn.Module):
    """
    HalfKAv2_hm Feature Transformer with shared weights.
    
    64 king buckets × 1280 features → 256 per bucket (shared weights)
    Accumulator size: 64 × 256 = 16,384
    """

    def __init__(
        self,
        num_buckets: int = 64,
        features_per_bucket: int = 1280,
        output_dim: int = 256,
        init_type: str = "kaiming_normal",
        init_gain: float = 1.0,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.features_per_bucket = features_per_bucket
        self.output_dim = output_dim
        self.input_dim = num_buckets * features_per_bucket

        # Shared linear layer across all buckets
        self.linear = nn.Linear(features_per_bucket, output_dim, bias=True)

        # Initialize with Kaiming Normal (critical for ReLU/ClippedReLU)
        self._init_weights(init_type, init_gain)

    def _init_weights(self, init_type: str, gain: float):
        if init_type == "kaiming_normal":
            nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='relu', a=math.sqrt(5))
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(self.linear.weight, gain=gain)
        elif init_type == "xavier_uniform":
            nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        else:
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)

        # Bias initialization
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 81920) or (batch, 64, 1280)
        Returns: (batch, 16384) accumulator
        """
        batch_size = x.shape[0]

        # Reshape to (batch, 64, 1280) for shared linear
        if x.dim() == 2:
            x = x.reshape(batch_size, self.num_buckets, self.features_per_bucket)

        # (batch, 64, 1280) -> (batch * 64, 1280)
        x_flat = x.reshape(-1, self.features_per_bucket)

        # Shared linear: (batch * 64, 256)
        acc = self.linear(x_flat)

        # ClippedReLU activation (torch.clamp is faster than custom module)
        acc = acc.clamp(min=0.0, max=1.0)

        # Reshape back to (batch, 64 * 256) = (batch, 16384)
        acc = acc.reshape(batch_size, -1)

        return acc


class NNUEModel(nn.Module):
    """
    Ultra-Optimized NNUE Network for Stockfish.
    
    Architecture:
    - FeatureTransformer: HalfKAv2_hm (shared weights, 64 buckets)
    - Hidden1: 16384 → 256 (ClippedReLU + Dropout)
    - Hidden2: 256 → 32 (ClippedReLU + Dropout)
    - Output: 32 → 1 (centipawns)
    
    Total params: ~2.5M trainable
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_cfg = config.get("model", {})
        self.clipped_relu_max = model_cfg.get("clipped_relu_max", 1.0)
        self.dropout_p = model_cfg.get("dropout", 0.0)

        # Feature transformer (shared weights)
        self.feature_transformer = FeatureTransformer(
            num_buckets=model_cfg.get("num_buckets", 64),
            features_per_bucket=model_cfg.get("features_per_bucket", 1280),
            output_dim=model_cfg.get("ft_output", 256),
            init_type=model_cfg.get("init_type", "kaiming_normal"),
            init_gain=model_cfg.get("init_gain", 1.0),
        )

        # Hidden layers - using separate Linear + clamp for torch.compile compatibility
        self.hidden1_linear = nn.Linear(
            model_cfg.get("ft_output", 256) * model_cfg.get("num_buckets", 64), 
            model_cfg.get("hidden1", 256)
        )
        self.hidden2_linear = nn.Linear(
            model_cfg.get("hidden1", 256), 
            model_cfg.get("hidden2", 32)
        )

        # Output layer (no activation - outputs centipawns)
        self.output_layer = nn.Linear(model_cfg.get("hidden2", 32), model_cfg.get("output_dim", 1))

        # Dropout layers for overfitting prevention
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)

        # Initialize hidden layers with Kaiming
        self._init_hidden_layers(model_cfg.get("init_type", "kaiming_normal"), model_cfg.get("init_gain", 1.0))

        # EMA model (for evaluation)
        self.ema_model: Optional[NNUEModel] = None
        self.ema_decay = config.get("advanced", {}).get("ema", {}).get("decay", 0.9999)

    def set_dropout(self, p: float):
        """Dynamically adjust dropout probability."""
        self.dropout_p = p
        self.dropout1.p = p
        self.dropout2.p = p
        print(f"Dropout set to {p}")

    def _init_hidden_layers(self, init_type: str, gain: float):
        for module in [self.hidden1_linear, self.hidden2_linear, self.output_layer]:
            if isinstance(module, nn.Linear):
                if init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                elif init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu', a=math.sqrt(5))
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(module.weight, gain=gain)
                elif init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 81920) dense features
        Returns: (batch, 1) centipawn evaluation
        """
        # Feature transformer: (batch, 81920) -> (batch, 16384)
        x = self.feature_transformer(x)

        # Hidden layers with inline ClippedReLU + Dropout (faster, torch.compile friendly)
        x = self.hidden1_linear(x).clamp(min=0.0, max=self.clipped_relu_max)  # (batch, 256)
        x = self.dropout1(x)
        x = self.hidden2_linear(x).clamp(min=0.0, max=self.clipped_relu_max)  # (batch, 32)
        x = self.dropout2(x)

        # Output
        x = self.output_layer(x)  # (batch, 1)

        return x

    def compile_model(self, mode: str = "max-autotune") -> "NNUEModel":
        """Compile model with torch.compile for speed.
        
        Returns the compiled model (caller must reassign):
            model = model.compile_model()
        """
        if hasattr(torch, 'compile'):
            try:
                # torch.compile returns a new _OrigMod wrapper; must return it
                compiled = torch.compile(self, mode=mode, fullgraph=False)
                print(f"Model compiled with torch.compile(mode={mode})")
                return compiled
            except Exception as e:
                print(f"torch.compile failed: {e}")
        return self

    def get_num_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def create_ema(self) -> "NNUEModel":
        """Create EMA copy of the model."""
        ema_model = NNUEModel(self.config)
        # Strip _orig_mod. prefix if self is a compiled model
        state = self.state_dict()
        clean_state = {}
        for k, v in state.items():
            clean_key = k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k
            clean_state[clean_key] = v
        ema_model.load_state_dict(clean_state)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        self.ema_model = ema_model
        return ema_model

    def update_ema(self, decay: Optional[float] = None):
        """Update EMA model parameters."""
        if self.ema_model is None:
            return

        decay = decay or self.ema_decay
        with torch.no_grad():
            ema_params = list(self.ema_model.parameters())
            # Get model parameters EXCLUDING ema_model submodule (to avoid double-counting)
            # since ema_model is a submodule of self, self.parameters() includes EMA params too
            params = []
            for name, param in self.named_parameters(recurse=True):
                if not name.startswith('ema_model.'):
                    params.append(param)
            assert len(ema_params) == len(params), f"EMA/model parameter count mismatch: {len(ema_params)} vs {len(params)}"
            for ema_param, param in zip(ema_params, params):
                ema_param.mul_(decay).add_(param, alpha=1.0 - decay)

    def forward_ema(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using EMA model."""
        if self.ema_model is None:
            return self.forward(x)
        self.ema_model.eval()
        with torch.no_grad():
            return self.ema_model(x)


def create_model(config: Dict[str, Any]) -> NNUEModel:
    """Factory function to create model from config."""
    return NNUEModel(config)