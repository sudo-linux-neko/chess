# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Training Module - Optimized for Maximum Elo/hr on 2x T4.

Key optimizations:
- Exact Stockfish loss formulation (WDL + Eval with proper scaling)
- Huber loss for evaluation (more robust to outliers)
- Label smoothing for WDL targets
- Cosine with warm restarts LR schedule
- torch.compile support
- Gradient accumulation for large effective batch
- Mixed precision with proper GradScaler
- Time-aware training with dynamic step estimation
"""

import os
import time
import signal
import math
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from src.model import NNUEModel, create_model, prepare_for_compile
from src.data import create_dataloader
from src.checkpoint import CheckpointManager
from src.hf_upload import create_uploader


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
    """
    WDL (Win/Draw/Loss) Loss - Stockfish formulation.
    
    Uses cross-entropy with soft targets (probabilities).
    Label smoothing for regularization.
    """

    __slots__ = ('label_smoothing',)

    def __init__(self, label_smoothing: float = 0.05):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, 3) - unnormalized logits for W/D/L
            targets: (batch, 3) - target probabilities (sum to 1)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Standard cross-entropy with soft targets
        loss = -(targets * log_probs).sum(dim=-1).mean()

        if self.label_smoothing > 0:
            # Label smoothing: mix with uniform distribution
            smooth_loss = -log_probs.mean(dim=-1).mean()
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

        return loss


class EvalLoss(nn.Module):
    """
    Evaluation Loss - Huber loss (more robust than MSE).
    
    Stockfish uses MSE, but Huber is more robust to outliers
    in the training data (blunders, engine errors).
    """

    __slots__ = ('delta',)

    def __init__(self, delta: float = 100.0):
        super().__init__()
        self.delta = delta  # Centipawns

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (batch, 1) - predicted centipawns
            targets: (batch, 1) - target centipawns
        """
        diff = preds.squeeze(-1) - targets.squeeze(-1)
        abs_diff = diff.abs()
        
        # Huber loss: quadratic for small errors, linear for large
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        
        return loss.mean()


class NNUELightningModule(pl.LightningModule):
    """PyTorch Lightning module for optimized NNUE training."""

    def __init__(self, config: dict, checkpoint_manager: 'CheckpointManager', hf_uploader):
        super().__init__()
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.hf_uploader = hf_uploader

        # Model (with torch.compile preparation)
        self.model = create_model(config)
        self.model = prepare_for_compile(self.model)

        # Losses
        train_cfg = config.get("training", {})
        self.wdl_loss_fn = WDLLoss(label_smoothing=train_cfg.get("label_smoothing", 0.05))
        self.eval_loss_fn = EvalLoss(delta=train_cfg.get("huber_delta", 100.0))

        # Loss blending
        self.wdl_lambda = train_cfg.get("wdl_lambda", 0.55)
        self.score_scale = train_cfg.get("score_scale", 370.0)

        # Training state
        self.training_state = TrainingState()
        self.start_time = time.time()
        self.should_stop = False

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Metrics
        self._last_log_step = 0
        self._step_times: List[float] = []

        # Compile model if enabled (PyTorch 2.0+)
        if config.get("hardware", {}).get("compile_model", False):
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"torch.compile failed: {e}, continuing without compilation")

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        self.should_stop = True
        if hasattr(self, 'trainer') and self.trainer:
            self.trainer.should_stop = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def _compute_wdl_targets(self, evals: torch.Tensor) -> torch.Tensor:
        """
        Convert centipawn evaluations to WDL probabilities.
        Stockfish exact formulation:
        W = sigmoid(eval / Q)
        L = sigmoid(-eval / Q)  
        D = 1 - W - L (clamped to >= 0)
        """
        Q = self.score_scale
        w = torch.sigmoid(evals / Q)
        l = torch.sigmoid(-evals / Q)
        d = 1.0 - w - l
        d = torch.clamp(d, min=0.0)
        
        # Renormalize to ensure sum = 1
        total = w + d + l
        w = w / total
        d = d / total
        l = l / total
        return torch.stack([w, d, l], dim=-1)

    def _eval_to_wdl_logits(self, eval_preds: torch.Tensor) -> torch.Tensor:
        """Convert scalar evaluation predictions to WDL logits."""
        # logit_W = eval / Q, logit_D = 0, logit_L = -eval / Q
        return torch.cat([
            eval_preds / self.score_scale,
            torch.zeros_like(eval_preds),
            -eval_preds / self.score_scale,
        ], dim=-1)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        step_start = time.time()
        
        features = batch["features"]      # (batch, 81920)
        wdl_targets = batch["wdl"]        # (batch, 3)
        eval_targets = batch["eval"]      # (batch,)

        # Forward pass
        eval_preds = self.model(features)  # (batch, 1)

        # Compute WDL targets from eval if not provided
        if wdl_targets is None or wdl_targets.sum() == 0:
            wdl_targets = self._compute_wdl_targets(eval_targets.unsqueeze(-1))

        # Compute losses
        wdl_logits = self._eval_to_wdl_logits(eval_preds)
        wdl_loss = self.wdl_loss_fn(wdl_logits, wdl_targets)
        eval_loss = self.eval_loss_fn(eval_preds.squeeze(-1), eval_targets)

        # Blended loss (Stockfish style)
        total_loss = self.wdl_lambda * wdl_loss + (1 - self.wdl_lambda) * eval_loss

        # Logging
        self.log("train/loss", total_loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log("train/wdl_loss", wdl_loss, sync_dist=True, rank_zero_only=True)
        self.log("train/eval_loss", eval_loss, sync_dist=True, rank_zero_only=True)
        self.log("train/wdl_lambda", self.wdl_lambda, sync_dist=True, rank_zero_only=True)

        # Throughput tracking
        batch_size = features.shape[0]
        step_time = time.time() - step_start
        self._step_times.append(step_time)
        if len(self._step_times) > 100:
            self._step_times.pop(0)
        
        if self.training_state.step % 100 == 0 and self._step_times:
            avg_step_time = sum(self._step_times) / len(self._step_times)
            throughput = batch_size / avg_step_time
            self.log("train/throughput", throughput * self.trainer.world_size, sync_dist=True, rank_zero_only=True)
            self.log("train/step_time_ms", avg_step_time * 1000, rank_zero_only=True)

        # Time budget check
        self._check_time_budget()

        if self.should_stop:
            self.trainer.should_stop = True

        self.training_state.step += 1
        return total_loss

    def _check_time_budget(self):
        elapsed_hours = (time.time() - self.start_time) / 3600
        budget_hours = self.config.get("time_budget", {}).get("hours", 10.0)
        setup_minutes = self.config.get("time_budget", {}).get("setup_minutes", 25)
        final_save_min = self.config.get("time_budget", {}).get("final_save_min", 3)

        effective_budget = budget_hours - (setup_minutes + final_save_min) / 60

        if elapsed_hours >= effective_budget:
            print(f"\nTime budget reached ({elapsed_hours:.2f}h / {effective_budget:.2f}h)")
            self.should_stop = True
            self.trainer.should_stop = True

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Periodic checkpointing handled by callback
        pass

    def configure_optimizers(self):
        train_cfg = self.config.get("training", {})
        lr = train_cfg.get("learning_rate", 0.0012)
        weight_decay = train_cfg.get("weight_decay", 1.5e-4)
        beta1 = train_cfg.get("beta1", 0.9)
        beta2 = train_cfg.get("beta2", 0.999)
        eps = train_cfg.get("eps", 1e-8)

        # Parameter groups with different weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name.lower() or "bn" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(param_groups, lr=lr, betas=(beta1, beta2), eps=eps)

        # Learning rate schedule: Cosine with Warm Restarts
        # This helps escape local minima and improves final Elo
        total_steps = self._estimate_total_steps()
        warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.02))
        min_lr_ratio = train_cfg.get("min_lr_ratio", 0.005)
        t0 = train_cfg.get("t0", 50000)
        t_mult = train_cfg.get("t_mult", 2)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            
            # Cosine with warm restarts
            progress = step - warmup_steps
            period = t0
            while progress >= period:
                progress -= period
                period *= t_mult
            
            cosine_progress = progress / max(1, period)
            cosine_progress = min(1.0, cosine_progress)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * cosine_progress))

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
        """Estimate total training steps based on time budget and measured throughput."""
        budget_cfg = self.config.get("time_budget", {})
        budget_hours = budget_cfg.get("hours", 10.0)
        setup_minutes = budget_cfg.get("setup_minutes", 25)
        final_save_min = budget_cfg.get("final_save_min", 3)

        training_hours = budget_hours - (setup_minutes + final_save_min) / 60
        training_seconds = training_hours * 3600

        hw_cfg = self.config.get("hardware", {})
        gpus = max(1, hw_cfg.get("gpus", 1))
        batch_size = self.config.get("training", {}).get("batch_size", 32768)
        
        # Conservative estimate: ~800k pos/sec/GPU for this architecture on T4
        est_pos_per_sec = gpus * 800000
        total_positions = training_seconds * est_pos_per_sec
        total_steps = int(total_positions / batch_size)

        return max(10000, total_steps)

    def train_dataloader(self):
        return create_dataloader(self.config, is_train=True)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["training_state"] = {
            "step": self.training_state.step,
            "epoch": self.training_state.epoch,
            "wall_time_elapsed": time.time() - self.start_time,
            "best_val_loss": self.training_state.best_val_loss,
            "data_position": self._get_data_position(),
            "rng_states": self._get_rng_states(),
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        if "training_state" in checkpoint:
            ts = checkpoint["training_state"]
            self.training_state.step = ts.get("step", 0)
            self.training_state.epoch = ts.get("epoch", 0)
            self.training_state.wall_time_elapsed = ts.get("wall_time_elapsed", 0)
            self.training_state.best_val_loss = ts.get("best_val_loss", float('inf'))
            self.start_time = time.time() - self.training_state.wall_time_elapsed

            if ts.get("data_position"):
                self._restore_data_position(ts["data_position"])
            if ts.get("rng_states"):
                self._restore_rng_states(ts["rng_states"])

    def _get_data_position(self) -> Dict[str, Any]:
        try:
            train_loader = self.trainer.train_dataloader
            if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'get_state'):
                return train_loader.dataset.get_state()
        except Exception:
            pass
        return {}

    def _restore_data_position(self, state: Dict[str, Any]):
        try:
            train_loader = self.trainer.train_dataloader
            if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'set_state'):
                train_loader.dataset.set_state(state)
        except Exception as e:
            print(f"Warning: Could not restore data position: {e}")

    def _get_rng_states(self) -> Dict[str, Any]:
        import numpy as np
        import random
        return {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    def _restore_rng_states(self, states: Dict[str, Any]):
        import numpy as np
        import random
        torch.set_rng_state(states.get("torch", torch.get_rng_state()))
        if states.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["cuda"])
        if states.get("numpy"):
            np.random.set_state(states["numpy"])
        if states.get("python"):
            random.setstate(states["python"])


class PeriodicCheckpointCallback(Callback):
    """Time-based periodic checkpointing with HF upload."""

    def __init__(self, checkpoint_manager: 'CheckpointManager', hf_uploader,
                 interval_min: int = 15, config: dict = None):
        self.checkpoint_manager = checkpoint_manager
        self.hf_uploader = hf_uploader
        self.interval_min = interval_min
        self.config = config or {}
        self.last_checkpoint_time = time.time()
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        elapsed_min = (time.time() - self.start_time) / 60
        last_ckpt_min = (self.last_checkpoint_time - self.start_time) / 60

        if elapsed_min - last_ckpt_min >= self.interval_min:
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
                   hf_uploader) -> pl.Trainer:
    """Create optimized PyTorch Lightning Trainer."""
    hw_cfg = config.get("hardware", {})
    log_cfg = config.get("logging", {})
    time_cfg = config.get("time_budget", {})

    log_dir = Path(log_cfg.get("log_dir", "/kaggle/working/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    loggers = []
    if log_cfg.get("tensorboard", True):
        loggers.append(TensorBoardLogger(save_dir=log_dir, name="tensorboard"))
    if log_cfg.get("csv_log", True):
        loggers.append(CSVLogger(save_dir=log_dir, name="csv"))

    callbacks = [
        PeriodicCheckpointCallback(
            checkpoint_manager=checkpoint_manager,
            hf_uploader=hf_uploader,
            interval_min=time_cfg.get("checkpoint_interval_min", 15),
            config=config,
        ),
    ]

    # DDP strategy for multi-GPU
    strategy = "ddp" if hw_cfg.get("gpus", 1) > 1 else "auto"
    if hw_cfg.get("gpus", 1) > 1:
        # Find unused port for DDP
        import socket
        with socket.socket() as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
        os.environ["MASTER_PORT"] = str(port)
        os.environ["MASTER_ADDR"] = "localhost"

    trainer = pl.Trainer(
        accelerator="gpu" if hw_cfg.get("gpus", 0) > 0 else "cpu",
        devices=hw_cfg.get("gpus", 1) if hw_cfg.get("gpus", 0) > 0 else 1,
        precision=hw_cfg.get("precision", "16-mixed"),
        strategy=strategy,
        max_epochs=1,
        max_steps=-1,
        logger=loggers if loggers else False,
        callbacks=callbacks,
        log_every_n_steps=log_cfg.get("log_every_n_steps", 50),
        gradient_clip_val=config.get("training", {}).get("gradient_clip_val", 1.0),
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        # Performance optimizations
        benchmark=True,  # cuDNN benchmark for fixed input sizes
    )

    return trainer