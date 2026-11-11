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
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.grads import grad_norm

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

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

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target: [W, D, L] probabilities or class indices
        if target.dim() == 2 and target.shape[1] == 3:
            # Soft targets - use KL divergence with label smoothing
            log_probs = F.log_softmax(logits, dim=-1)
            # Apply label smoothing to target distribution
            if self.label_smoothing > 0:
                target = target * (1 - self.label_smoothing) + self.label_smoothing / 3
                target = target / target.sum(dim=-1, keepdim=True)
            loss = -(target * log_probs).sum(dim=-1).mean()
        else:
            # Hard targets
            loss = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
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
        self._global_start_time = time.time()
        self._epoch_start_time = time.time()
        
        # Time budget
        time_cfg = config.get("time_budget", {})
        self.max_time_seconds = time_cfg.get("hours", 9.5) * 3600
        self.checkpoint_interval = time_cfg.get("checkpoint_interval_min", 10) * 60
        self._last_checkpoint_time = time.time()
        
        # RNG for lambda jitter (worker-safe)
        self._lambda_rng = torch.Generator()
        self._lambda_rng.manual_seed(config.get("seed", 42))
        
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
        """Single training step. Gradient accumulation handled by PL via accumulate_grad_batches."""
        features = batch["features"].to(self.device, non_blocking=True)
        wdl_target = batch["wdl"].to(self.device, non_blocking=True)
        eval_target = batch["eval"].to(self.device, non_blocking=True)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.get("advanced", {}).get("amp", {}).get("enabled", True)):
            eval_pred = self.model(features)
            
            # WDL logits from eval
            wdl_logits = self._eval_to_wdl_logits(eval_pred)
            
            # Compute lambda for this step
            lambda_val = self._get_lambda()
            
            # Losses
            loss_wdl = self.wdl_loss_fn(wdl_logits, wdl_target)
            loss_eval = self.eval_loss_fn(eval_pred.squeeze(), eval_target)
            
            # Combined loss
            loss = lambda_val * loss_wdl + (1.0 - lambda_val) * loss_eval
        
        # Scale loss for gradient accumulation (PL does NOT auto-scale)
        loss = loss / self.accum_steps
        
        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite loss detected: {loss.item()}")
            loss = torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)
        
        # Log losses (only on accumulation boundaries to reduce noise)
        if (batch_idx + 1) % self.accum_steps == 0:
            # Log UNSCALED loss for monitoring (actual loss value)
            self.log("train/loss_wdl", loss_wdl.detach(), prog_bar=True, sync_dist=True)
            self.log("train/loss_eval", loss_eval.detach(), prog_bar=True, sync_dist=True)
            self.log("train/loss_total", (loss.detach() * self.accum_steps), prog_bar=True, sync_dist=True)
            self.log("train/lambda", lambda_val, prog_bar=True, sync_dist=True)
            
            self.training_state.train_losses.append((loss.detach() * self.accum_steps).item())
            self.training_state.lambda_history.append(lambda_val)
        
        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int):
        """Handle end of accumulation step - update EMA, checkpoint, time check."""
        # Only run on accumulation boundaries (PL calls this after optimizer step)
        if (batch_idx + 1) % self.accum_steps != 0:
            return
            
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
        
        # Checkpointing
        self._maybe_checkpoint()
        
        # Time budget check
        if self._check_time_budget():
            self.trainer.should_stop = True

    def _eval_to_wdl_logits(self, eval_cp: torch.Tensor) -> torch.Tensor:
        """Convert centipawn evaluation to WDL logits."""
        # Stockfish-style conversion using sigmoid
        # W = sigmoid(eval / scale), L = sigmoid(-eval / scale), D = 1 - W - L
        scale = self.config.get("training", {}).get("wdl_scale", 400.0)
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
        
        # Add jitter using worker-safe RNG
        if self.lambda_jitter.get("enabled", True):
            sample_jitter = self.lambda_jitter.get("sample_jitter", 0.0035)
            batch_jitter = self.lambda_jitter.get("batch_jitter", 0.0070)
            decay = self.lambda_jitter.get("decay", 0.999)
            
            # Per-sample and per-batch jitter using torch RNG
            # Cap decay factor to prevent underflow for large step values
            decay_factor = min(decay ** min(step, 100000), 1.0)
            jitter = (torch.rand(1, generator=self._lambda_rng).item() * 2 - 1) * sample_jitter
            jitter += (torch.rand(1, generator=self._lambda_rng).item() * 2 - 1) * batch_jitter * decay_factor
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
                ema_model=self.ema_model,  # Pass EMA model for checkpointing
            )
            if ckpt_path:
                self.hf_uploader.queue_upload(ckpt_path)
        except Exception as e:
            print(f"Checkpoint save failed: {e}")

    def _check_time_budget(self) -> bool:
        """Check if time budget exceeded."""
        elapsed = time.time() - self._global_start_time + self.training_state.time_elapsed
        return elapsed >= self.max_time_seconds

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step using EMA model."""
        features = batch["features"]
        wdl_target = batch["wdl"]
        eval_target = batch["eval"]
        
        # Use EMA model for validation
        model = self.ema_model if self.ema_model is not None else self.model
        was_training = model.training
        model.eval()
        
        try:
            with torch.inference_mode():
                eval_pred = model(features)
                wdl_logits = self._eval_to_wdl_logits(eval_pred)
                
                # Use the same lambda as training (already computed for this step)
                lambda_val = self._get_lambda()
                loss_wdl = self.wdl_loss_fn(wdl_logits, wdl_target)
                loss_eval = self.eval_loss_fn(eval_pred.squeeze(), eval_target)
                loss = lambda_val * loss_wdl + (1.0 - lambda_val) * loss_eval
        finally:
            # Restore training mode
            model.train(was_training)
        
        self.log("val/loss_wdl", loss_wdl, prog_bar=True, sync_dist=True)
        self.log("val/loss_eval", loss_eval, prog_bar=True, sync_dist=True)
        self.log("val/loss_total", loss, prog_bar=True, sync_dist=True)
        
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
        # Use fused AdamW if available (CUDA only, faster)
        optimizer_kwargs = {
            "lr": train_cfg.get("learning_rate", 0.0008),
            "weight_decay": train_cfg.get("weight_decay", 4e-5),
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }
        try:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                fused=True,
                **optimizer_kwargs,
            )
        except TypeError:
            # fused not supported (CPU or older PyTorch)
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                **optimizer_kwargs,
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

    def on_before_optimizer_step(self, optimizer):
        """Log learning rate and apply gradient clipping before optimizer step."""
        # Log current LR for progress bar (every optimizer step)
        lr = optimizer.param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, sync_dist=True)
        
        # Gradient clipping (PL does this automatically but we add explicit safety)
        grad_clip_val = self.config.get("training", {}).get("gradient_clip_val", 1.0)
        if grad_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_val)

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
            "time_elapsed": self.training_state.time_elapsed + (time.time() - self._global_start_time),
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

    def on_train_end(self):
        """Cleanup CUDA memory at end of training."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class TimeLimitCallback(Callback):
    """Callback to enforce time limit."""
    
    def __init__(self, max_hours: float = 9.5):
        self.max_seconds = max_hours * 3600
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            print(f"Time budget ({self.max_seconds/3600:.1f}h) exceeded. Stopping...")
            trainer.should_stop = True


class TQDMProgressBar(Callback):
    """Custom tqdm progress bar with ETA, steps/sec, and detailed metrics."""
    
    def __init__(self, max_steps: int, refresh_rate: int = 1):
        super().__init__()
        self.max_steps = max_steps
        self.refresh_rate = refresh_rate
        self.pbar = None
        self.start_time = None
        self.last_update_time = None
        self.last_step = 0
        
    def on_train_start(self, trainer, pl_module):
        if not TQDM_AVAILABLE:
            return
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_step = 0
        
        # Get current step (for resume)
        current_step = getattr(pl_module, 'training_state', None)
        initial = current_step.step if current_step else 0
        
        self.pbar = tqdm(
            total=self.max_steps,
            initial=initial,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not TQDM_AVAILABLE or self.pbar is None:
            return
            
        current_step = trainer.global_step
        
        # Only update every refresh_rate steps
        if current_step % self.refresh_rate != 0 and current_step != self.max_steps:
            return
            
        now = time.time()
        elapsed = now - self.start_time
        step_diff = current_step - self.last_step
        time_diff = now - self.last_update_time
        
        if time_diff > 0:
            steps_per_sec = step_diff / time_diff
        else:
            steps_per_sec = 0
        
        # Calculate ETA
        remaining_steps = self.max_steps - current_step
        if steps_per_sec > 0:
            eta_seconds = remaining_steps / steps_per_sec
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "∞"
        
        # Get metrics
        metrics = trainer.callback_metrics
        loss = metrics.get("train/loss_total", None)
        lr = metrics.get("lr", None)
        lambda_val = metrics.get("train/lambda", None)
        
        # Update progress bar
        postfix = {
            "loss": f"{loss:.4f}" if loss is not None else "N/A",
            "lr": f"{lr:.2e}" if lr is not None else "N/A",
            "λ": f"{lambda_val:.3f}" if lambda_val is not None else "N/A",
            "steps/s": f"{steps_per_sec:.1f}",
            "ETA": eta_str,
        }
        
        self.pbar.set_postfix(postfix)
        self.pbar.update(current_step - self.pbar.n)
        
        self.last_step = current_step
        self.last_update_time = now
    
    def on_train_end(self, trainer, pl_module):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}m{s:02d}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h{m:02d}m"


def create_trainer(config: Dict[str, Any], checkpoint_manager: CheckpointManager, hf_uploader: HFUploader) -> pl.Trainer:
    """Create PyTorch Lightning trainer with ultra-optimized settings."""
    hw_cfg = config.get("hardware", {})
    time_cfg = config.get("time_budget", {})
    log_cfg = config.get("logging", {})
    adv_cfg = config.get("advanced", {})
    
    max_steps = config.get("training", {}).get("lr_schedule", {}).get("total_steps", 5108000)
    
    # Callbacks - TimeLimitCallback handles time budget; TQDMProgressBar shows progress
    callbacks = [
        TimeLimitCallback(
            max_hours=time_cfg.get("hours", 9.5),
        ),
        TQDMProgressBar(
            max_steps=max_steps,
            refresh_rate=log_cfg.get("progress_bar_refresh_rate", 10),
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
    
    # Validation interval: val_check_interval can be int (epochs) or float (fraction of epoch)
    val_check_interval = adv_cfg.get("validation", {}).get("val_check_interval", 0.05)
    if isinstance(val_check_interval, float):
        check_val_every_n_epoch = 1
        limit_val_batches = val_check_interval
    else:
        check_val_every_n_epoch = val_check_interval
        limit_val_batches = 1.0
    
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=hw_cfg.get("gpus", 2),
        strategy="ddp_find_unused_parameters_false" if hw_cfg.get("gpus", 2) > 1 else "auto",
        precision=precision,
        max_epochs=1000,  # Limited by time/steps
        max_steps=config.get("training", {}).get("lr_schedule", {}).get("total_steps", 5108000),
        gradient_clip_val=config.get("training", {}).get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=config.get("training", {}).get("gradient_clip_algorithm", "norm"),
        accumulate_grad_batches=adv_cfg.get("gradient_accumulation", 8),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=log_cfg.get("log_every_n_steps", 100),
        enable_checkpointing=False,  # We handle checkpointing manually in module
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        benchmark=hw_cfg.get("benchmark_cudnn", True),
        num_sanity_val_steps=0,
        check_val_every_n_epoch=check_val_every_n_epoch,
        limit_val_batches=limit_val_batches,
    )
    
    return trainer