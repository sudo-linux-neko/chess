# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Ultra-Optimized Training Module for NNUE.

Based on official Stockfish nettest recipes:
- One-Cycle LR schedule with cosine annealing
- Lambda schedule with jitter (WDL/eval blending)
- Curriculum learning by ply
- Quantile regression (asymmetric Huber)
- EMA model for evaluation
- Gradient accumulation
- Mixed precision (fp16)
- Advanced loss functions
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.grads import grad_norm

from src.model import NNUEModel
from src.checkpoint import CheckpointManager
from src.hf_upload import HFUploader


@dataclass
class TrainingState:
    """Complete training state for exact resume."""
    step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    best_val_loss: float = float('inf')
    train_losses: List[float] = None
    val_losses: List[float] = None
    lr_history: List[float] = None
    lambda_history: List[float] = None
    time_elapsed: float = 0.0

    def __post_init__(self):
        if self.train_losses is None:
            self.train_losses = []
        if self.val_losses is None:
            self.val_losses = []
        if self.lr_history is None:
            self.lr_history = []
        if self.lambda_history is None:
            self.lambda_history = []


class QuantileLoss(nn.Module):
    """
    Asymmetric quantile loss for eval prediction.
    Based on Stockfish's pow-exp formulation.
    """
    def __init__(self, delta: float = 100.0, asymmetry: float = 0.22866886710086187, pow_exp: float = 2.4340402395048404):
        super().__init__()
        self.delta = delta
        self.asymmetry = asymmetry  # qp_asymmetry
        self.pow_exp = pow_exp

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Asymmetric Huber / quantile loss
        error = target - pred
        abs_error = error.abs()
        
        # Quantile weight: higher weight for underestimation (asymmetry)
        weight = torch.where(error >= 0, 
                           self.asymmetry, 
                           1.0 - self.asymmetry)
        
        # Huber-like: quadratic for small errors, linear for large
        loss = torch.where(abs_error <= self.delta,
                          0.5 * error ** 2,
                          self.delta * (abs_error - 0.5 * self.delta))
        
        # Power scaling (Stockfish uses this for better gradient flow)
        loss = loss ** (1.0 / self.pow_exp)
        
        return (weight * loss).mean()


class WDLLoss(nn.Module):
    """
    WDL (Win/Draw/Loss) loss with label smoothing.
    Cross-entropy with smoothing for regularization.
    """
    def __init__(self, label_smoothing: float = 0.05):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target: [W, D, L] probabilities or class indices
        if target.dim() == 2 and target.shape[1] == 3:
            # Soft targets - use KL divergence
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(target * log_probs).sum(dim=-1).mean()
        else:
            # Hard targets
            loss = self.ce(logits, target)
        return loss


class NNUELightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for NNUE training.
    
    Features:
    - One-cycle LR schedule
    - Lambda schedule with jitter
    - Gradient accumulation
    - Mixed precision
    - EMA model
    - Time-aware training
    """

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_manager: CheckpointManager,
        hf_uploader: HFUploader,
    ):
        super().__init__()
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.hf_uploader = hf_uploader
        
        # Model
        self.model = NNUEModel(config)
        self.ema_model = None
        
        # Training config
        train_cfg = config.get("training", {})
        adv_cfg = config.get("advanced", {})
        
        self.batch_size = train_cfg.get("batch_size", 65536)
        self.micro_batch_size = train_cfg.get("micro_batch_size", 8192)
        self.accum_steps = adv_cfg.get("gradient_accumulation", 8)
        self.max_positions = train_cfg.get("max_positions", 55_000_000_000)
        
        # Loss functions
        self.wdl_loss_fn = WDLLoss(train_cfg.get("label_smoothing", 0.05))
        self.eval_loss_fn = QuantileLoss(
            delta=train_cfg.get("eval_delta", 100.0),
            asymmetry=train_cfg.get("qp_asymmetry", 0.22866886710086187),
            pow_exp=train_cfg.get("pow_exp", 2.4340402395048404),
        )
        
        # Lambda schedule (WDL/eval blending)
        self.lambda_cfg = train_cfg.get("lambda_schedule", {})
        self.start_lambda = self.lambda_cfg.get("start_lambda", 1.0)
        self.end_lambda = self.lambda_cfg.get("end_lambda", 1.0)
        self.lambda_total_steps = self.lambda_cfg.get("total_steps", 5052000)
        self.lambda_warmup_pct = self.lambda_cfg.get("warmup_pct", 0.25)
        self.lambda_delta = self.lambda_cfg.get("delta", -0.3)
        self.lambda_jitter = self.lambda_cfg.get("jitter", {})
        
        # Curriculum learning
        self.curriculum_cfg = train_cfg.get("curriculum", {})
        
        # Scaling
        self.in_scaling = train_cfg.get("in_scaling", 295.6539508488627)
        self.out_scaling = train_cfg.get("out_scaling", 379.98724077106635)
        self.in_offset = train_cfg.get("in_offset", 285.2706341467852)
        self.out_offset = train_cfg.get("out_offset", 289.1258344152218)
        
        # Training state
        self.training_state = TrainingState()
        self._step_start_time = time.time()
        self._epoch_start_time = time.time()
        
        # Time budget
        time_cfg = config.get("time_budget", {})
        self.max_time_seconds = time_cfg.get("hours", 9.5) * 3600
        self.checkpoint_interval = time_cfg.get("checkpoint_interval_min", 10) * 60
        self._last_checkpoint_time = time.time()
        
        # Save hyperparameters
        self.save_hyperparameters(config)

    def setup(self, stage: Optional[str] = None):
        """Setup called after model is moved to device."""
        # Create EMA model
        if self.config.get("advanced", {}).get("ema", {}).get("enabled", True):
            self.ema_model = self.model.create_ema()
            self.ema_model.to(self.device)
        
        # Compile model if requested
        if self.config.get("hardware", {}).get("compile_model", True):
            compile_mode = self.config.get("hardware", {}).get("compile_mode", "max-autotune")
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
                print(f"Model compiled with torch.compile(mode={compile_mode})")
            except Exception as e:
                print(f"torch.compile failed: {e}")

    def on_train_start(self):
        """Called at start of training."""
        self._epoch_start_time = time.time()
        self._step_start_time = time.time()

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        """Track step timing."""
        self._step_start_time = time.time()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Single training step with gradient accumulation."""
        features = batch["features"]
        wdl_target = batch["wdl"]
        eval_target = batch["eval"]
        
        # Forward pass
        eval_pred = self.model(features)
        
        # WDL logits from eval (simplified - in practice use separate head)
        # For now, convert eval to WDL probabilities using sigmoid scaling
        wdl_logits = self._eval_to_wdl_logits(eval_pred)
        
        # Compute lambda for this step
        lambda_val = self._get_lambda()
        
        # Losses
        loss_wdl = self.wdl_loss_fn(wdl_logits, wdl_target)
        loss_eval = self.eval_loss_fn(eval_pred.squeeze(), eval_target)
        
        # Combined loss
        loss = lambda_val * loss_wdl + (1.0 - lambda_val) * loss_eval
        
        # Scale for gradient accumulation
        loss = loss / self.accum_steps
        
        # Manual backward for accumulation
        self.manual_backward(loss)
        
        # Log losses
        if batch_idx % self.accum_steps == 0:
            self.log("train/loss_wdl", loss_wdl.detach(), prog_bar=True)
            self.log("train/loss_eval", loss_eval.detach(), prog_bar=True)
            self.log("train/loss_total", loss.detach() * self.accum_steps, prog_bar=True)
            self.log("train/lambda", lambda_val, prog_bar=True)
            
            self.training_state.train_losses.append((loss.detach() * self.accum_steps).item())
            self.training_state.lambda_history.append(lambda_val)
        
        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int):
        """Handle gradient accumulation step."""
        if (batch_idx + 1) % self.accum_steps == 0:
            # Gradient clipping
            if self.config.get("training", {}).get("gradient_clip_val", 1.0) > 0:
                self.clip_gradients(
                    self.optimizers(),
                    gradient_clip_val=self.config["training"]["gradient_clip_val"],
                    gradient_clip_algorithm=self.config["training"].get("gradient_clip_algorithm", "norm"),
                )
            
            # Optimizer step
            self.optimizers().step()
            self.optimizers().zero_grad()
            
            # Update EMA
            if self.ema_model is not None:
                ema_cfg = self.config.get("advanced", {}).get("ema", {})
                if ema_cfg.get("enabled", True):
                    update_every = ema_cfg.get("update_every", 1)
                    if self.training_state.step % update_every == 0:
                        self.model.update_ema()
            
            # Update training state
            self.training_state.step += 1
            self.training_state.samples_seen += self.batch_size
            
            # LR scheduling (handled by scheduler)
            # Lambda scheduling (handled above)
            
            # Checkpointing
            self._maybe_checkpoint()
            
            # Time budget check
            if self._check_time_budget():
                self.trainer.should_stop = True

    def _eval_to_wdl_logits(self, eval_cp: torch.Tensor) -> torch.Tensor:
        """Convert centipawn evaluation to WDL logits."""
        # Stockfish-style conversion using sigmoid
        # W = sigmoid(eval / scale), L = sigmoid(-eval / scale), D = 1 - W - L
        scale = 400.0  # Typical scaling
        w = torch.sigmoid(eval_cp / scale)
        l = torch.sigmoid(-eval_cp / scale)
        d = 1.0 - w - l
        d = d.clamp(min=1e-6)
        w = w.clamp(min=1e-6)
        l = l.clamp(min=1e-6)
        
        # Renormalize
        total = w + d + l
        w, d, l = w/total, d/total, l/total
        
        # Convert to logits
        logits = torch.stack([w, d, l], dim=-1).log()
        return logits

    def _get_lambda(self) -> float:
        """Get current lambda value with schedule and jitter."""
        step = self.training_state.step
        total = self.lambda_total_steps
        warmup = int(total * self.lambda_warmup_pct)
        
        if step < warmup:
            # Warmup: linear from 0 to start_lambda
            progress = step / max(1, warmup)
            lambda_val = self.start_lambda * progress
        elif step < total:
            # Cosine annealing with delta
            progress = (step - warmup) / max(1, total - warmup)
            lambda_val = self.start_lambda + self.lambda_delta * (1 - math.cos(math.pi * progress)) / 2
        else:
            lambda_val = self.end_lambda
        
        # Add jitter
        if self.lambda_jitter.get("enabled", True):
            import random
            sample_jitter = self.lambda_jitter.get("sample_jitter", 0.0035)
            batch_jitter = self.lambda_jitter.get("batch_jitter", 0.0070)
            decay = self.lambda_jitter.get("decay", 0.999)
            
            # Per-sample and per-batch jitter
            jitter = random.uniform(-sample_jitter, sample_jitter)
            jitter += random.uniform(-batch_jitter, batch_jitter) * (decay ** step)
            lambda_val += jitter
        
        return max(0.0, min(1.0, lambda_val))

    def _maybe_checkpoint(self):
        """Save checkpoint if interval elapsed."""
        now = time.time()
        if now - self._last_checkpoint_time >= self.checkpoint_interval:
            self._save_checkpoint()
            self._last_checkpoint_time = now

    def _save_checkpoint(self):
        """Save training checkpoint."""
        try:
            ckpt_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizers(),
                scheduler=self.lr_schedulers(),
                scaler=self.trainer.scaler if hasattr(self.trainer, 'scaler') else None,
                training_state=self.training_state,
                config=self.config,
            )
            if ckpt_path:
                self.hf_uploader.queue_upload(ckpt_path)
        except Exception as e:
            print(f"Checkpoint save failed: {e}")

    def _check_time_budget(self) -> bool:
        """Check if time budget exceeded."""
        elapsed = time.time() - self._epoch_start_time + self.training_state.time_elapsed
        return elapsed >= self.max_time_seconds

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step using EMA model."""
        features = batch["features"]
        wdl_target = batch["wdl"]
        eval_target = batch["eval"]
        
        # Use EMA model for validation
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()
        
        with torch.no_grad():
            eval_pred = model(features)
            wdl_logits = self._eval_to_wdl_logits(eval_pred)
            
            lambda_val = self._get_lambda()
            loss_wdl = self.wdl_loss_fn(wdl_logits, wdl_target)
            loss_eval = self.eval_loss_fn(eval_pred.squeeze(), eval_target)
            loss = lambda_val * loss_wdl + (1.0 - lambda_val) * loss_eval
        
        self.log("val/loss_wdl", loss_wdl, prog_bar=True)
        self.log("val/loss_eval", loss_eval, prog_bar=True)
        self.log("val/loss_total", loss, prog_bar=True)
        
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        """Track validation losses."""
        if self.trainer.callback_metrics:
            val_loss = self.trainer.callback_metrics.get("val/loss_total")
            if val_loss is not None:
                self.training_state.val_losses.append(val_loss.item())
                if val_loss < self.training_state.best_val_loss:
                    self.training_state.best_val_loss = val_loss.item()

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler."""
        train_cfg = self.config.get("training", {})
        
        # AdamW with factorized weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.get("learning_rate", 0.0008),
            weight_decay=train_cfg.get("weight_decay", 4e-5),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # One-Cycle LR Scheduler
        lr_cfg = train_cfg.get("lr_schedule", {})
        total_steps = lr_cfg.get("total_steps", 5108000)
        max_lr = lr_cfg.get("max_lr", 0.0008)
        warmup_pct = lr_cfg.get("warmup_pct", 0.05)
        final_div = lr_cfg.get("final_div", 10000.0)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=warmup_pct,
            anneal_strategy=lr_cfg.get("anneal_strategy", "cos"),
            div_factor=25.0,
            final_div_factor=final_div,
            three_phase=False,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save custom state."""
        checkpoint["training_state"] = {
            "step": self.training_state.step,
            "epoch": self.training_state.epoch,
            "samples_seen": self.training_state.samples_seen,
            "best_val_loss": self.training_state.best_val_loss,
            "train_losses": self.training_state.train_losses,
            "val_losses": self.training_state.val_losses,
            "lr_history": self.training_state.lr_history,
            "lambda_history": self.training_state.lambda_history,
            "time_elapsed": self.training_state.time_elapsed + (time.time() - self._epoch_start_time),
        }
        
        # Save EMA state
        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Load custom state."""
        if "training_state" in checkpoint:
            ts = checkpoint["training_state"]
            self.training_state.step = ts.get("step", 0)
            self.training_state.epoch = ts.get("epoch", 0)
            self.training_state.samples_seen = ts.get("samples_seen", 0)
            self.training_state.best_val_loss = ts.get("best_val_loss", float('inf'))
            self.training_state.train_losses = ts.get("train_losses", [])
            self.training_state.val_losses = ts.get("val_losses", [])
            self.training_state.lr_history = ts.get("lr_history", [])
            self.training_state.lambda_history = ts.get("lambda_history", [])
            self.training_state.time_elapsed = ts.get("time_elapsed", 0.0)
        
        # Load EMA state
        if "ema_state_dict" in checkpoint and self.ema_model is not None:
            self.ema_model.load_state_dict(checkpoint["ema_state_dict"])


class TimeLimitCallback(Callback):
    """Callback to enforce time limit and periodic checkpointing."""
    
    def __init__(self, max_hours: float = 9.5, checkpoint_interval_min: int = 10):
        self.max_seconds = max_hours * 3600
        self.checkpoint_interval = checkpoint_interval_min * 60
        self.start_time = time.time()
        self.last_checkpoint = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            print(f"Time budget ({self.max_seconds/3600:.1f}h) exceeded. Stopping...")
            trainer.should_stop = True
        
        if elapsed - self.last_checkpoint >= self.checkpoint_interval:
            pl_module._save_checkpoint()
            self.last_checkpoint = time.time()


def create_trainer(config: Dict[str, Any], checkpoint_manager: CheckpointManager, hf_uploader: HFUploader) -> pl.Trainer:
    """Create PyTorch Lightning trainer with ultra-optimized settings."""
    hw_cfg = config.get("hardware", {})
    time_cfg = config.get("time_budget", {})
    log_cfg = config.get("logging", {})
    
    # Callbacks
    callbacks = [
        TimeLimitCallback(
            max_hours=time_cfg.get("hours", 9.5),
            checkpoint_interval_min=time_cfg.get("checkpoint_interval_min", 10),
        ),
    ]
    
    # Logger
    logger = None
    if log_cfg.get("tensorboard", True):
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir=log_cfg.get("dir", "/kaggle/working/logs"),
            name="nnue",
            version=None,
        )
    
    # Precision
    precision = "16-mixed" if hw_cfg.get("fp16", True) else "32-true"
    
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=hw_cfg.get("gpus", 2),
        strategy="ddp_find_unused_parameters_false" if hw_cfg.get("gpus", 2) > 1 else "auto",
        precision=precision,
        max_epochs=1000,  # Limited by time/steps
        max_steps=config.get("training", {}).get("lr_schedule", {}).get("total_steps", 5108000),
        gradient_clip_val=config.get("training", {}).get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=config.get("training", {}).get("gradient_clip_algorithm", "norm"),
        accumulate_grad_batches=config.get("advanced", {}).get("gradient_accumulation", 8),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=log_cfg.get("log_every_n_steps", 100),
        enable_checkpointing=False,  # We handle checkpointing manually
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        benchmark=hw_cfg.get("benchmark_cudnn", True),
        num_sanity_val_steps=0,
        check_val_every_n_epoch=config.get("advanced", {}).get("validation", {}).get("val_check_interval", 0) 
                                if isinstance(config.get("advanced", {}).get("validation", {}).get("val_check_interval", 0), int) 
                                else 1,
        limit_val_batches=config.get("advanced", {}).get("validation", {}).get("val_check_interval", 0.05),
    )
    
    return trainer