# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Training Module using PyTorch Lightning.

Implements the training loop, loss functions, LR scheduling,
and integration with checkpointing/HF upload.
"""

import os
import time
import signal
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Try to import Ranger optimizer
try:
    from ranger import Ranger
    RANGER_AVAILABLE = True
except ImportError:
    RANGER_AVAILABLE = False
    Ranger = None

from src.model import NNUEModel, create_model
from src.data import create_dataloader, StreamingBinpackDataset
from src.checkpoint import CheckpointManager
from src.hf_upload import HFUploader


@dataclass
class TrainingState:
    """Complete training state for checkpointing."""
    step: int = 0
    epoch: int = 0
    wall_time_elapsed: float = 0.0
    best_val_loss: float = float('inf')
    data_position: Dict[str, Any] = None
    rng_states: Dict[str, Any] = None


class WDLLoss(nn.Module):
    """WDL (Win/Draw/Loss) loss with label smoothing."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, 3) - unnormalized logits for W/D/L
            targets: (batch, 3) - target probabilities (sum to 1)
        """
        # Convert targets to log-probabilities for cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()

        if self.label_smoothing > 0:
            # Uniform smoothing
            smooth_loss = -log_probs.mean(dim=-1).mean()
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

        return loss


class EvalLoss(nn.Module):
    """MSE loss for centipawn evaluation."""

    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (batch, 1) - predicted centipawns
            targets: (batch, 1) - target centipawns
        """
        return F.mse_loss(preds.squeeze(-1), targets.squeeze(-1))


class NNUELightningModule(pl.LightningModule):
    """PyTorch Lightning module for NNUE training."""

    def __init__(self, config: dict, checkpoint_manager: 'CheckpointManager', hf_uploader: 'HFUploader'):
        super().__init__()
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.hf_uploader = hf_uploader

        # Model
        self.model = create_model(config)

        # Losses
        self.wdl_loss_fn = WDLLoss(label_smoothing=0.0)
        self.eval_loss_fn = EvalLoss()

        # Loss blending
        self.wdl_lambda = config.get("training", {}).get("wdl_lambda", 0.5)
        self.score_scale = config.get("training", {}).get("score_scale", 370.0)

        # Training state
        self.training_state = TrainingState()
        self.start_time = time.time()
        self.should_stop = False

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Metrics tracking
        self.train_losses: List[float] = []
        self.train_wdl_losses: List[float] = []
        self.train_eval_losses: List[float] = []
        self.throughput_history: List[float] = []

        # Automatic optimization (we handle backward manually for mixed precision)
        self.automatic_optimization = True

    def _signal_handler(self, signum, frame):
        """Handle SIGTERM/SIGINT for graceful checkpoint save."""
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        self.should_stop = True
        self.trainer.should_stop = True

    def forward(self, features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Forward pass - features are sparse indices, weights are values."""
        # For HalfKAv2, we need to convert sparse indices to dense
        # The C++ loader returns dense features already (one-hot encoded per bucket)
        # But to be efficient, we'll assume the data loader provides dense tensors
        return self.model(features)

    def _compute_wdl_targets(self, evals: torch.Tensor) -> torch.Tensor:
        """
        Convert centipawn evaluations to WDL probabilities using Stockfish sigmoid.
        W = 1 / (1 + exp(-eval/Q)), L = 1 - W, D = 1 - W - L (approximated)
        Actually Stockfish uses: W = sigmoid(eval/Q), L = sigmoid(-eval/Q), D = 1 - W - L
        """
        Q = self.score_scale
        w = torch.sigmoid(evals / Q)
        l = torch.sigmoid(-evals / Q)
        d = 1.0 - w - l
        # Clamp to valid probabilities
        d = torch.clamp(d, 0.0, 1.0)
        # Renormalize
        total = w + d + l
        w = w / total
        d = d / total
        l = l / total
        return torch.stack([w, d, l], dim=-1)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        features = batch["features"]      # (batch, 81920) - dense
        weights = batch["weights"]        # (batch, 81920) - usually all 1.0
        wdl_targets = batch["wdl"]        # (batch, 3)
        eval_targets = batch["eval"]      # (batch,)

        # Forward pass
        eval_preds = self.model(features)  # (batch, 1)

        # Compute WDL targets from eval if not provided
        # In binpack, WDL targets are usually provided directly
        if wdl_targets is None or wdl_targets.sum() == 0:
            wdl_targets = self._compute_wdl_targets(eval_targets.unsqueeze(-1))

        # Compute losses
        # For WDL loss, we need logits - but our model outputs scalar eval
        # We convert eval_preds to WDL logits using the same sigmoid
        wdl_logits = torch.cat([
            eval_preds / self.score_scale,      # W logit
            torch.zeros_like(eval_preds),       # D logit (implicit)
            -eval_preds / self.score_scale,     # L logit
        ], dim=-1)

        wdl_loss = self.wdl_loss_fn(wdl_logits, wdl_targets)
        eval_loss = self.eval_loss_fn(eval_preds.squeeze(-1), eval_targets)

        # Blended loss
        total_loss = self.wdl_lambda * wdl_loss + (1 - self.wdl_lambda) * eval_loss

        # Logging
        self.log("train/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("train/wdl_loss", wdl_loss, sync_dist=True)
        self.log("train/eval_loss", eval_loss, sync_dist=True)
        self.log("train/wdl_lambda", self.wdl_lambda, sync_dist=True)

        # Track metrics
        self.train_losses.append(total_loss.item())
        self.train_wdl_losses.append(wdl_loss.item())
        self.train_eval_losses.append(eval_loss.item())

        # Throughput tracking
        batch_size = features.shape[0]
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            throughput = (self.training_state.step * batch_size) / elapsed
            self.throughput_history.append(throughput)
            self.log("train/throughput", throughput, sync_dist=True)

        # Check for time budget
        self._check_time_budget()

        # Check for graceful stop
        if self.should_stop:
            self.trainer.should_stop = True

        self.training_state.step += 1
        return total_loss

    def _check_time_budget(self):
        """Check if we've exceeded time budget, trigger checkpoint if needed."""
        elapsed_hours = (time.time() - self.start_time) / 3600
        budget_hours = self.config.get("time_budget", {}).get("hours", 10.0)
        setup_minutes = self.config.get("time_budget", {}).get("setup_minutes", 30)
        final_save_min = self.config.get("time_budget", {}).get("final_save_min", 5)

        effective_budget = budget_hours - (setup_minutes + final_save_min) / 60

        if elapsed_hours >= effective_budget:
            print(f"\nTime budget reached ({elapsed_hours:.2f}h / {effective_budget:.2f}h)")
            self.should_stop = True
            self.trainer.should_stop = True

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Check for periodic checkpointing after each batch."""
        interval_min = self.config.get("time_budget", {}).get("checkpoint_interval_min", 20)
        elapsed_min = (time.time() - self.start_time) / 60

        # Check if it's time to checkpoint (approximately)
        if hasattr(self, '_last_checkpoint_min'):
            if elapsed_min - self._last_checkpoint_min >= interval_min:
                self._save_periodic_checkpoint()
                self._last_checkpoint_min = elapsed_min
        else:
            self._last_checkpoint_min = elapsed_min

    def _save_periodic_checkpoint(self):
        """Save periodic checkpoint and queue for HF upload."""
        if self.trainer.is_global_zero:
            print(f"\n[Step {self.training_state.step}] Saving periodic checkpoint...")
            ckpt_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizers(),
                scheduler=self.lr_schedulers(),
                scaler=self.scaler if hasattr(self, 'scaler') else None,
                training_state=self.training_state,
                config=self.config,
            )
            # Queue for background HF upload
            if ckpt_path and self.hf_uploader:
                self.hf_uploader.queue_upload(ckpt_path)

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler."""
        train_cfg = self.config.get("training", {})
        lr = train_cfg.get("learning_rate", 8e-4)
        weight_decay = train_cfg.get("weight_decay", 1e-4)
        optimizer_name = train_cfg.get("optimizer", "ranger").lower()

        # Separate weight decay for different parameter groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if optimizer_name == "ranger" and RANGER_AVAILABLE:
            optimizer = Ranger(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        else:
            optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)

        # Time-aware cosine scheduler
        total_steps = self._estimate_total_steps()
        warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.01))
        min_lr_ratio = train_cfg.get("min_lr_ratio", 0.01)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            # Cosine decay
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, progress)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _estimate_total_steps(self) -> int:
        """Estimate total training steps based on time budget."""
        budget_cfg = self.config.get("time_budget", {})
        budget_hours = budget_cfg.get("hours", 10.0)
        setup_minutes = budget_cfg.get("setup_minutes", 30)
        final_save_min = budget_cfg.get("final_save_min", 5)

        training_hours = budget_hours - (setup_minutes + final_save_min) / 60
        training_seconds = training_hours * 3600

        # Estimate positions per second (will be calibrated during training)
        hw_cfg = self.config.get("hardware", {})
        gpus = max(1, hw_cfg.get("gpus", 1))
        batch_size = self.config.get("training", {}).get("batch_size", 16384)
        est_pos_per_sec = gpus * 600000  # Conservative estimate

        total_positions = training_seconds * est_pos_per_sec
        total_steps = int(total_positions / batch_size)

        return max(1000, total_steps)  # Minimum steps

    def train_dataloader(self):
        """Create training dataloader."""
        return create_dataloader(self.config, is_train=True)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Called by Lightning when saving checkpoint."""
        checkpoint["training_state"] = {
            "step": self.training_state.step,
            "epoch": self.training_state.epoch,
            "wall_time_elapsed": time.time() - self.start_time,
            "best_val_loss": self.training_state.best_val_loss,
            "data_position": self._get_data_position(),
            "rng_states": self._get_rng_states(),
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Called by Lightning when loading checkpoint."""
        if "training_state" in checkpoint:
            ts = checkpoint["training_state"]
            self.training_state.step = ts.get("step", 0)
            self.training_state.epoch = ts.get("epoch", 0)
            self.training_state.wall_time_elapsed = ts.get("wall_time_elapsed", 0)
            self.training_state.best_val_loss = ts.get("best_val_loss", float('inf'))
            self.start_time = time.time() - self.training_state.wall_time_elapsed

            # Restore data position
            if ts.get("data_position"):
                self._restore_data_position(ts["data_position"])

            # Restore RNG states
            if ts.get("rng_states"):
                self._restore_rng_states(ts["rng_states"])

    def _get_data_position(self) -> Dict[str, Any]:
        """Get current data loader position for resume."""
        # Access the dataset through the dataloader
        try:
            train_loader = self.trainer.train_dataloader
            if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'get_state'):
                return train_loader.dataset.get_state()
        except Exception:
            pass
        return {}

    def _restore_data_position(self, state: Dict[str, Any]):
        """Restore data loader position."""
        try:
            train_loader = self.trainer.train_dataloader
            if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'set_state'):
                train_loader.dataset.set_state(state)
        except Exception as e:
            print(f"Warning: Could not restore data position: {e}")

    def _get_rng_states(self) -> Dict[str, Any]:
        """Get all RNG states for exact reproducibility."""
        return {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    def _restore_rng_states(self, states: Dict[str, Any]):
        """Restore all RNG states."""
        torch.set_rng_state(states.get("torch", torch.get_rng_state()))
        if states.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["cuda"])
        if states.get("numpy"):
            np.random.set_state(states["numpy"])
        if states.get("python"):
            random.setstate(states["python"])


class PeriodicCheckpointCallback(Callback):
    """Callback for time-based periodic checkpointing."""

    def __init__(self, checkpoint_manager: 'CheckpointManager', hf_uploader: 'HFUploader',
                 interval_min: int = 20, config: dict = None):
        self.checkpoint_manager = checkpoint_manager
        self.hf_uploader = hf_uploader
        self.interval_min = interval_min
        self.config = config or {}
        self.last_checkpoint_time = time.time()
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapsed_min = (time.time() - self.start_time) / 60
        last_ckpt_min = (self.last_checkpoint_time - self.start_time) / 60

        if elapsed_min - last_ckpt_min >= self.interval_min:
            if trainer.is_global_zero:
                print(f"\n[Step {pl_module.training_state.step}] Periodic checkpoint...")
                ckpt_path = self.checkpoint_manager.save_checkpoint(
                    model=pl_module.model,
                    optimizer=pl_module.optimizers(),
                    scheduler=pl_module.lr_schedulers(),
                    scaler=getattr(pl_module, 'scaler', None),
                    training_state=pl_module.training_state,
                    config=self.config,
                )
                if ckpt_path and self.hf_uploader:
                    self.hf_uploader.queue_upload(ckpt_path)
                self.last_checkpoint_time = time.time()


def create_trainer(config: dict, checkpoint_manager: 'CheckpointManager',
                   hf_uploader: 'HFUploader') -> pl.Trainer:
    """Create PyTorch Lightning Trainer with all callbacks."""
    hw_cfg = config.get("hardware", {})
    log_cfg = config.get("logging", {})
    time_cfg = config.get("time_budget", {})

    # Loggers
    log_dir = Path(log_cfg.get("log_dir", "/kaggle/working/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    loggers = []
    if log_cfg.get("tensorboard", True):
        loggers.append(TensorBoardLogger(save_dir=log_dir, name="tensorboard"))
    if log_cfg.get("csv_log", True):
        loggers.append(CSVLogger(save_dir=log_dir, name="csv"))

    # Callbacks
    callbacks = [
        PeriodicCheckpointCallback(
            checkpoint_manager=checkpoint_manager,
            hf_uploader=hf_uploader,
            interval_min=time_cfg.get("checkpoint_interval_min", 20),
            config=config,
        ),
    ]

    # Trainer config
    trainer = pl.Trainer(
        accelerator="gpu" if hw_cfg.get("gpus", 0) > 0 else "cpu",
        devices=hw_cfg.get("gpus", 1) if hw_cfg.get("gpus", 0) > 0 else 1,
        precision=hw_cfg.get("precision", "16-mixed"),
        strategy="ddp" if hw_cfg.get("gpus", 1) > 1 else "auto",
        max_epochs=1,  # We control steps via time budget
        max_steps=-1,  # Unlimited, controlled by time budget
        logger=loggers if loggers else False,
        callbacks=callbacks,
        log_every_n_steps=log_cfg.get("log_every_n_steps", 100),
        gradient_clip_val=config.get("training", {}).get("gradient_clip_val", 1.0),
        enable_checkpointing=False,  # We handle checkpointing manually
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=0,
    )

    return trainer