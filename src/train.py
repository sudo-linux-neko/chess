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
import os
import time
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

from src.model import NNUEModel
from src.checkpoint import CheckpointManager
from src.hf_upload import HFUploader
from src.data import create_dataloader


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
    # Data resume position
    current_file_idx: int = 0
    current_position: int = 0

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
        streaming_manager: Optional[Any] = None,
    ):
        super().__init__()
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.hf_uploader = hf_uploader
        self.streaming_manager = streaming_manager
        
        # Model
        self.model = NNUEModel(config)
        self.ema_model = None
        
        # Training config
        train_cfg = config.get("training", {})
        adv_cfg = config.get("advanced", {})
        
        self.batch_size = train_cfg.get("batch_size", 65536)
        self.micro_batch_size = train_cfg.get("micro_batch_size", 8192)
        # Support both naming conventions: "advanced.gradient_accumulation" (train_ultra.yaml)
        # and "training.accumulate_grad_batches" (train.yaml). The latter is the PL-standard
        # key and takes precedence if both are set.
        self.accum_steps = adv_cfg.get("gradient_accumulation", train_cfg.get("accumulate_grad_batches", 8))
        self.max_positions = train_cfg.get("max_positions", 55_000_000_000)
        
        # Loss functions
        self.wdl_loss_fn = WDLLoss(float(train_cfg.get("label_smoothing", 0.05)))
        self.eval_loss_fn = QuantileLoss(
            delta=float(train_cfg.get("eval_delta", 100.0)),
            asymmetry=float(train_cfg.get("qp_asymmetry", 0.22866886710086187)),
            pow_exp=float(train_cfg.get("pow_exp", 2.4340402395048404)),
        )
        
        # Lambda schedule (WDL/eval blending)
        self.lambda_cfg = train_cfg.get("lambda_schedule", {})
        # Default start_lambda to wdl_lambda from training config (not 1.0)
        # so the lambda schedule starts at the intended blending ratio.
        # Use explicit None check to handle start_lambda: null in YAML.
        _start_lambda = self.lambda_cfg.get("start_lambda", train_cfg.get("wdl_lambda", 1.0))
        if _start_lambda is None:
            _start_lambda = train_cfg.get("wdl_lambda", 1.0)
        self.start_lambda = float(_start_lambda)

        _lambda_delta = self.lambda_cfg.get("delta", -0.3)
        if _lambda_delta is None:
            _lambda_delta = -0.3
        self.lambda_delta = float(_lambda_delta)

        # end_lambda defaults to start_lambda + delta so the schedule
        # is continuous (no jump after total_steps)
        _end_lambda = self.lambda_cfg.get("end_lambda", self.start_lambda + self.lambda_delta)
        if _end_lambda is None:
            _end_lambda = self.start_lambda + self.lambda_delta
        self.end_lambda = float(_end_lambda)
        # Support both "T0" (Stockfish terminology) and "total_steps" config keys
        self.lambda_total_steps = int(self.lambda_cfg.get("total_steps", self.lambda_cfg.get("T0", 5052000)))
        self.lambda_warmup_pct = float(self.lambda_cfg.get("warmup_pct", 0.25))
        self.lambda_jitter = self.lambda_cfg.get("jitter", {})
        if self.lambda_jitter is None:
            self.lambda_jitter = {}
        
        # Curriculum learning
        self.curriculum_cfg = train_cfg.get("curriculum", {})
        
        # Scaling
        self.in_scaling = float(train_cfg.get("in_scaling", 295.6539508488627))
        self.out_scaling = float(train_cfg.get("out_scaling", 379.98724077106635))
        self.in_offset = float(train_cfg.get("in_offset", 285.2706341467852))
        self.out_offset = float(train_cfg.get("out_offset", 289.1258344152218))
        
        # Training state
        self.training_state = TrainingState()
        self._global_start_time = time.time()
        self._epoch_start_time = time.time()
        
        # Resume time offset: previously elapsed time before this process started
        self._resume_time_offset = 0.0
        
        # Resume step offset: PL's trainer.global_step starts at 0 on fresh Trainer,
        # but we need training_state.step to continue from checkpoint value.
        self._resume_step_offset = 0
        
        # Samples seen at resume time - used to correctly estimate data position
        # after resume (samples_seen includes pre-resume samples, so we need the
        # delta to estimate post-resume position)
        self._samples_seen_at_resume = 0
        
        # Estimated positions per file (for approximate data position tracking on resume)
        # Populated lazily on first _update_data_position() call
        self._positions_per_file_estimate = None
        
        # Flag to ensure train_dataloader() only uses resume position on first call
        self._first_train_dataloader = True
        
        # Time budget
        time_cfg = config.get("time_budget", {})
        self.max_time_seconds = time_cfg.get("hours", 9.5) * 3600
        self.checkpoint_interval = time_cfg.get("checkpoint_interval_min", 10) * 60
        self._last_checkpoint_time = time.time()
        self._last_global_step = -1  # Track PL global step for accurate accumulation boundaries
        self._last_log_step = -1  # Track last logged global step to avoid batch_idx misalignment
        
        # RNG for lambda jitter (worker-safe, seeded per-rank for DDP diversity)
        self._lambda_rng = torch.Generator()
        self._lambda_rng.manual_seed(config.get("seed", 42))

        # Reseed lambda jitter based on global rank so each DDP process
        # gets different jitter values (called after trainer is assigned)
        self._lambda_rank_seeded = False
        
        # Save hyperparameters
        self.save_hyperparameters(config)

    def setup(self, stage: Optional[str] = None):
        """Setup called after model is moved to device."""
        # Seed lambda jitter RNG per DDP rank for jitter diversity
        if not getattr(self, '_lambda_rank_seeded', False):
            try:
                rank = self.trainer.global_rank
            except Exception:
                rank = 0
            # Restore saved RNG state from checkpoint if resuming (exact lambda jitter continuity)
            _lambda_rng_state = getattr(self, '_lambda_rng_state', None)
            if _lambda_rng_state is not None:
                try:
                    self._lambda_rng.set_state(_lambda_rng_state)
                except Exception:
                    self._lambda_rng.manual_seed(self.config.get("seed", 42) + rank * 10000)
            else:
                self._lambda_rng.manual_seed(self.config.get("seed", 42) + rank * 10000)
            self._lambda_rank_seeded = True
        
        # Enable TF32 for faster matmul on T4 (Ampere+)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        
        # Create EMA model
        if self.config.get("advanced", {}).get("ema", {}).get("enabled", True):
            self.ema_model = self.model.create_ema()
            self.ema_model.to(self.device)
            # Restore EMA state from checkpoint on resume
            _ema_sd = getattr(self, '_ema_state_dict', None)
            if _ema_sd:
                try:
                    self.ema_model.load_state_dict(_ema_sd)
                    print("  EMA state restored from checkpoint")
                except Exception as e:
                    print(f"  Warning: Could not restore EMA state: {e}")
        
        # Compile model if requested
        if self.config.get("hardware", {}).get("compile_model", True):
            compile_mode = self.config.get("hardware", {}).get("compile_mode", "max-autotune")
            try:
                # fullgraph=False for robustness - allows Python fallbacks
                self.model = torch.compile(self.model, mode=compile_mode, fullgraph=False)
                print(f"Model compiled with torch.compile(mode={compile_mode})")
            except Exception as e:
                print(f"torch.compile failed: {e}")

    def on_train_start(self):
        """Called at start of training."""
        self._epoch_start_time = time.time()
        self._step_start_time = time.time()
        
        # Restore optimizer/scheduler state from checkpoint.
        # PL creates new optimizers in configure_optimizers() during fit(),
        # so manually-loaded state (before fit) is lost. We store state dicts
        # on the module before fit() and restore them here, after PL has
        # created its own optimizers but before the training loop starts.
        if getattr(self, '_optimizer_state', None):
            try:
                optimizers = self.optimizers()
                if optimizers and len(optimizers) > 0:
                    optimizers[0].load_state_dict(self._optimizer_state)
                    print("  Optimizer state restored from checkpoint")
            except Exception as e:
                print(f"  Warning: Could not restore optimizer state: {e}")
        
        if getattr(self, '_scheduler_state', None):
            try:
                scheds = self.lr_schedulers()
                if scheds and len(scheds) > 0:
                    scheds[0].load_state_dict(self._scheduler_state)
                    print("  Scheduler state restored from checkpoint")
            except Exception as e:
                print(f"  Warning: Could not restore scheduler state: {e}")
        
        # Restore EMA state from checkpoint (setup() created the EMA model,
        # now we overwrite it with the checkpointed EMA weights)
        if getattr(self, '_ema_state_dict', None) and self.ema_model is not None:
            try:
                # Strip _orig_mod. prefix if present in checkpointed EMA state
                clean_ema_state = {}
                for k, v in self._ema_state_dict.items():
                    clean_key = k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k
                    clean_ema_state[clean_key] = v
                self.ema_model.load_state_dict(clean_ema_state)
                print("  EMA state restored from checkpoint")
            except Exception as e:
                print(f"  Warning: Could not restore EMA state: {e}")

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        """Track step timing."""
        self._step_start_time = time.time()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Single training step. Gradient accumulation handled by PL via accumulate_grad_batches.
        
        NOTE: Do NOT manually divide loss by accum_steps - PyTorch Lightning 2.x
        automatically scales loss for gradient accumulation. Manual division would
        cause double-division, making gradients accum_steps^2 too small.
        """
        # Sync training_state.step with PL's global_step (accounts for resume offset).
        # This must happen BEFORE _get_lambda() so lambda uses the correct step.
        # on_train_batch_end() also updates this (idempotent), but training_step
        # runs first and _get_lambda() needs the current value.
        self.training_state.step = self.trainer.global_step + self._resume_step_offset
        
        features = batch["features"].to(self.device, non_blocking=True)
        wdl_target = batch["wdl"].to(self.device, non_blocking=True)
        eval_target = batch["eval"].to(self.device, non_blocking=True)
        
        # Forward pass - PL handles AMP automatically via precision="16-mixed"
        eval_pred = self.model(features)
        
        # WDL logits from eval
        wdl_logits = self._eval_to_wdl_logits(eval_pred)
        
        # Compute lambda for this step
        lambda_val = self._get_lambda()
        
        # Losses
        loss_wdl = self.wdl_loss_fn(wdl_logits, wdl_target)
        loss_eval = self.eval_loss_fn(eval_pred.squeeze(-1), eval_target)
        
        # Combined loss (unscaled - PL handles accumulation scaling)
        loss = lambda_val * loss_wdl + (1.0 - lambda_val) * loss_eval
        
        # Check for NaN/Inf - skip step if non-finite to avoid corrupting weights
        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite loss detected at step {self.training_state.step}, skipping step")
            # Return clean zero loss disconnected from graph - NaN * 0 = NaN, so
            # we must NOT multiply by loss. A fresh zero tensor prevents weight corruption.
            return torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=True)
        
        # Log losses (only on accumulation boundaries to reduce noise)
        # Use trainer.global_step instead of batch_idx % accum_steps because
        # batch_idx resets each epoch while PL's accumulation counter is global.
        current_step = self.trainer.global_step
        if current_step == self._last_log_step:
            pass  # Already logged this step
        elif (current_step - self._last_log_step) >= 1:
            self._last_log_step = current_step
            self.log("train/loss_wdl", loss_wdl.detach(), prog_bar=True, sync_dist=True)
            self.log("train/loss_eval", loss_eval.detach(), prog_bar=True, sync_dist=True)
            self.log("train/loss_total", loss.detach(), prog_bar=True, sync_dist=True)
            self.log("train/lambda", lambda_val, prog_bar=True, sync_dist=True)
            
            self.training_state.train_losses.append(loss.detach().item())
            self.training_state.lambda_history.append(lambda_val)
        
        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int):
        """Handle end of accumulation step - update EMA, checkpoint, time check."""
        # Only run on optimizer step boundaries.
        # Use trainer.global_step instead of batch_idx % accum_steps because
        # batch_idx resets each epoch while PL's accumulation counter is global.
        # When epoch length doesn't divide evenly by accum_steps, batch_idx-based
        # detection misaligns with PL's actual optimizer steps.
        current_step = self.trainer.global_step
        if current_step == self._last_global_step:
            return
        self._last_global_step = current_step
        
        # Sync training_state.step with PL's global_step, accounting for resume offset.
        # PL's trainer.global_step starts at 0 on a fresh Trainer instance, but after
        # resume we need step to continue from the checkpoint value (e.g., step N).
        # So: training_state.step = current_step + _resume_step_offset.
        self.training_state.step = current_step + self._resume_step_offset
        
        # Update EMA
        if self.ema_model is not None:
            ema_cfg = self.config.get("advanced", {}).get("ema", {})
            if ema_cfg.get("enabled", True):
                update_every = ema_cfg.get("update_every", 1)
                if update_every > 0 and self.training_state.step % update_every == 0:
                    # Access underlying model (torch.compile wraps it in _orig_mod)
                    model_for_ema = self.model
                    if hasattr(model_for_ema, '_orig_mod'):
                        model_for_ema = model_for_ema._orig_mod
                    model_for_ema.update_ema()
        
        # Update training state - samples_seen is per-process count (not multiplied
        # by world_size) because _update_data_position() estimates per-process file
        # position from it. Multiplying by world_size would inflate the count and
        # cause wrong file/position tracking on resume.
        #
        # NOTE: on_train_batch_end only runs on optimizer step boundaries
        # (guard: current_step != _last_global_step), so each call represents
        # accum_steps batches × micro_batch_size samples = micro_batch_size * accum_steps.
        self.training_state.samples_seen += self.micro_batch_size * self.accum_steps
        
        # Track approximate data position for resume (computed in main process)
        # Dataset position updates in worker processes don't propagate to main process,
        # so we estimate from samples_seen using positions-per-file from first file.
        self._update_data_position()
        
        # Checkpointing
        self._maybe_checkpoint()
        
        # Max positions check - stop training when enough positions processed
        # samples_seen is per-rank; multiply by world_size for global budget
        world_size = getattr(self.trainer, 'world_size', 1) or 1
        global_samples = self.training_state.samples_seen * world_size
        if self.max_positions and global_samples >= self.max_positions:
            print(f"\n  Max positions ({self.max_positions:,}) reached at step {self.training_state.step}, stopping training")
            self.trainer.should_stop = True
        
        # Time budget check
        if self._check_time_budget():
            self.trainer.should_stop = True

    def _eval_to_wdl_logits(self, eval_cp: torch.Tensor) -> torch.Tensor:
        """Convert centipawn evaluation to WDL logits."""
        # Stockfish-style conversion using sigmoid
        # W = sigmoid(eval / scale), L = sigmoid(-eval / scale), D = 1 - W - L
        scale = float(self.config.get("training", {}).get("wdl_scale", 400.0))
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
            
            # Single RNG call for both jitter components (faster)
            # Cap decay factor to prevent underflow for large step values
            decay_factor = min(decay ** min(step, 100000), 1.0)
            jitter_vals = torch.rand(2, generator=self._lambda_rng)
            jitter = (jitter_vals[0].item() * 2 - 1) * sample_jitter
            jitter += (jitter_vals[1].item() * 2 - 1) * batch_jitter * decay_factor
            lambda_val += jitter
        
        return max(0.0, min(1.0, lambda_val))

    def _maybe_checkpoint(self):
        """Save checkpoint if interval elapsed (rank 0 only in DDP)."""
        if self.trainer.global_rank != 0:
            return
        now = time.time()
        if now - self._last_checkpoint_time >= self.checkpoint_interval:
            self._save_checkpoint()
            self._last_checkpoint_time = now

    def _save_checkpoint(self):
        """Save training checkpoint (rank 0 only)."""
        if self.trainer.global_rank != 0:
            return
        try:
            # Sync data position from training_state (computed in on_train_batch_end
            # from samples_seen). Do NOT read from self.train_dataset because
            # dataset position updates happen in worker processes (separate memory
            # space with spawn multiprocessing) and never propagate to main process.
            # The training_state values are always current.
            
            # Update cumulative time_elapsed before saving
            self.training_state.time_elapsed = time.time() - self._global_start_time + self._resume_time_offset
            
            # Get optimizer/scheduler with protection (may not be available before trainer.fit() completes setup)
            try:
                optimizers = self.optimizers()
                opt = optimizers[0] if optimizers else None
            except Exception:
                opt = None
            try:
                scheds = self.lr_schedulers()
                sched = scheds[0] if scheds else None
            except Exception:
                sched = None
            
            ckpt_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=opt,
                scheduler=sched,
                scaler=self.trainer.scaler if hasattr(self.trainer, 'scaler') and self.trainer.scaler is not None else None,
                training_state=self.training_state,
                config=self.config,
                ema_model=self.ema_model,  # Pass EMA model for checkpointing
                data_state={"_lambda_rng": self._lambda_rng.get_state()},  # Preserve lambda jitter RNG across resume
            )
            if ckpt_path and self.hf_uploader:
                try:
                    self.hf_uploader.queue_upload(ckpt_path)
                    # Also upload metadata JSON with loss, time, step, etc.
                    meta_path = ckpt_path.with_suffix('.meta.json')
                    if meta_path.exists():
                        self.hf_uploader.queue_upload(
                            meta_path,
                            remote_path=f"checkpoints/{meta_path.name}",
                            priority=5,
                        )
                except Exception as e:
                    print(f"Checkpoint upload failed: {e}")
        except Exception as e:
            print(f"Checkpoint save failed: {e}")
            return

    def _check_time_budget(self) -> bool:
        """Check if time budget exceeded."""
        elapsed = time.time() - self._global_start_time + self._resume_time_offset
        return elapsed >= self.max_time_seconds

    def _update_data_position(self):
        """Estimate current data file/position from samples_seen for resume.
        
        Dataset position updates happen in worker processes (separate memory space
        with spawn multiprocessing) and never propagate to main process. We estimate
        position from samples_seen using positions-per-file from the first file.
        
        Under DDP with stride sharding (files[rank::world_size]), each rank processes
        a disjoint subset of files. Each rank tracks its own samples_seen, and we
        convert per-rank samples to global file index by:
          global_file = base_file + rank + shard_files_completed * world_size
        """
        dataset = getattr(self, 'train_dataset', None)
        if dataset is None or not hasattr(dataset, 'files') or len(dataset.files) == 0:
            return
        
        # Lazily estimate positions per file from first file
        if self._positions_per_file_estimate is None:
            try:
                first_file = dataset.files[0]
                from src.data import CPP_LOADER_AVAILABLE, CppBinpackDataset
                if CPP_LOADER_AVAILABLE:
                    ds = CppBinpackDataset(str(first_file))
                    self._positions_per_file_estimate = len(ds)
                else:
                    import struct
                    with open(first_file, 'rb') as f:
                        f.read(4)  # magic
                        f.read(4)  # version
                        self._positions_per_file_estimate = struct.unpack("<Q", f.read(8))[0]
            except Exception:
                self._positions_per_file_estimate = 100000  # fallback
        
        ppf = self._positions_per_file_estimate
        if ppf <= 0:
            return
        
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        
        num_files = len(dataset.files)
        
        # Per-rank samples since resume (each rank tracks its own samples_seen).
        # Do NOT multiply by world_size - each rank processes its own shard
        # independently, so per-rank samples map directly to per-rank file progress.
        samples_since_resume = self.training_state.samples_seen - self._samples_seen_at_resume
        if samples_since_resume < 0:
            samples_since_resume = 0
        
        base_file = getattr(self, '_resume_start_file', 0)
        base_position = getattr(self, '_resume_start_position', 0)
        samples_remaining_in_first_file = ppf - base_position
        if samples_since_resume <= samples_remaining_in_first_file:
            # Still in the first (resume-start) file
            files_completed = 0
            position_in_file = base_position + samples_since_resume
        else:
            # Moved past the first file
            remaining = samples_since_resume - samples_remaining_in_first_file
            files_completed = 1 + (remaining // ppf)
            position_in_file = remaining % ppf
        
        # Convert per-rank shard position to global file index.
        # base_file is rank 0's saved global index. Each rank r starts at
        # base_file + r (its first file in the stride shard). Each rank's
        # shard files advance the global index by files_completed * world_size.
        global_file_idx = base_file + rank + files_completed * world_size
        
        self.training_state.current_file_idx = min(global_file_idx, num_files - 1)
        self.training_state.current_position = min(position_in_file, ppf - 1)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step using EMA model."""
        features = batch["features"].to(self.device, non_blocking=True)
        wdl_target = batch["wdl"].to(self.device, non_blocking=True)
        eval_target = batch["eval"].to(self.device, non_blocking=True)
        
        # Use EMA model for validation
        model = self.ema_model if self.ema_model is not None else self.model
        was_training = model.training
        model.eval()
        
        # Save RNG state so validation doesn't affect training lambda jitter
        lambda_rng_state = self._lambda_rng.get_state()
        
        try:
            with torch.inference_mode():
                eval_pred = model(features)
                wdl_logits = self._eval_to_wdl_logits(eval_pred)
                
                # Use a FIXED lambda for validation (don't consume training RNG jitter)
                lambda_val = self.start_lambda
                loss_wdl = self.wdl_loss_fn(wdl_logits, wdl_target)
                loss_eval = self.eval_loss_fn(eval_pred.squeeze(-1), eval_target)
                loss = lambda_val * loss_wdl + (1.0 - lambda_val) * loss_eval
        finally:
            # Restore training mode and RNG state
            model.train(was_training)
            self._lambda_rng.set_state(lambda_rng_state)
        
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

    def train_dataloader(self):
        """Return training dataloader.
        
        Only the first call after setup/resume uses the checkpoint resume
        position. Subsequent epochs (epoch 2+) must start from file 0, position 0
        for normal multi-epoch training. Otherwise epoch 2+ would start from the
        position where epoch 1 ended (mid-file), causing data to be skipped or
        re-read incorrectly.
        """
        # Only use resume position on the very first dataloader creation
        # (right after checkpoint resume). For subsequent epochs, start fresh.
        use_resume_pos = getattr(self, '_first_train_dataloader', True)
        self._first_train_dataloader = False
        
        if use_resume_pos:
            start_file = self.training_state.current_file_idx
            start_position = self.training_state.current_position
            # current_file_idx is already a global index (includes rank offset
            # from _update_data_position: base_file + rank + files_completed *
            # world_size). Do NOT add rank again — that would double-count and
            # skip a file on non-zero ranks.
            # Track samples_seen at resume so _update_data_position() can correctly
            # estimate post-resume position (samples_seen includes pre-resume samples)
            self._samples_seen_at_resume = self.training_state.samples_seen
            self._resume_start_file = start_file
            self._resume_start_position = start_position
        else:
            start_file = 0
            start_position = 0
            # Reset resume base for new epoch so _update_data_position() estimates
            # from the beginning of this epoch (not from the original resume point).
            # Without this, stale _resume_start_file/_resume_start_position from a
            # prior checkpoint resume would corrupt position estimation on epoch 2+.
            self._resume_start_file = 0
            self._resume_start_position = 0
            self._samples_seen_at_resume = self.training_state.samples_seen
        
        loader = create_dataloader(
            self.config,
            streaming_manager=self.streaming_manager,
            is_train=True,
            start_file=start_file,
            start_position=start_position,
        )
        # Reset resume position after first use
        self.training_state.current_file_idx = 0
        self.training_state.current_position = 0
        # Store dataset reference so we can read its position during checkpoint saves
        self.train_dataset = loader.dataset
        return loader

    def val_dataloader(self):
        """Return validation dataloader (uses a small subset for fast validation)."""
        # Create a separate validation dataloader with fewer files for speed
        # Use deepcopy to avoid mutating self.config["data"] via shallow-copy aliasing
        import copy
        val_config = copy.deepcopy(self.config)
        val_cfg = val_config.get("data", {})
        val_cfg["num_files_simultaneous"] = min(val_cfg.get("num_files_simultaneous", 6), 2)
        return create_dataloader(
            val_config,
            streaming_manager=self.streaming_manager,
            is_train=False,
        )

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler."""
        train_cfg = self.config.get("training", {})
        
        # Cast config values to float (YAML can load scientific notation as string)
        lr = float(train_cfg.get("learning_rate", 0.0008))
        weight_decay = float(train_cfg.get("weight_decay", 4e-5))
        betas_raw = train_cfg.get("betas", (0.9, 0.999))
        if not isinstance(betas_raw, (list, tuple)):
            betas_raw = (float(betas_raw), 0.999)
        # Ensure exactly 2 betas (AdamW requires beta1, beta2)
        betas_list = [float(b) for b in betas_raw]
        if len(betas_list) == 0:
            betas_list = [0.9, 0.999]
        elif len(betas_list) == 1:
            betas_list = [betas_list[0], 0.999]
        betas = tuple(betas_list[:2])  # Take only first 2 if more provided
        eps = float(train_cfg.get("eps", 1e-8))
        
        # AdamW with factorized weight decay
        # Use fused AdamW if available (CUDA only, faster).
        # Fused optimizers don't support gradient clipping in PyTorch Lightning,
        # so disable fused when gradient clipping is configured.
        #
        # IMPORTANT: Exclude ema_model submodule parameters from the optimizer.
        # ema_model is registered as a submodule of NNUEModel, so self.model.parameters()
        # would include EMA params, which would bloat checkpoints with optimizer state
        # for EMA params and could corrupt EMA if requires_grad ever changes.
        optimizer_kwargs = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        grad_clip = self.config.get("training", {}).get("gradient_clip_val", 0.0)
        use_fused = torch.cuda.is_available() and grad_clip == 0.0

        # Get model parameters excluding EMA submodule (same filtering as update_ema)
        # Handle both compiled (_orig_mod.ema_model.xxx) and uncompiled (ema_model.xxx) names
        def _get_trainable_params():
            params = []
            for name, param in self.model.named_parameters(recurse=True):
                if name.startswith('ema_model.') or name.startswith('_orig_mod.ema_model.'):
                    continue
                params.append(param)
            return params

        if use_fused:
            try:
                optimizer = torch.optim.AdamW(
                    _get_trainable_params(),
                    fused=True,
                    **optimizer_kwargs,
                )
            except (TypeError, RuntimeError, ValueError):
                # fused not supported (CPU or older PyTorch/CUDA)
                optimizer = torch.optim.AdamW(
                    _get_trainable_params(),
                    **optimizer_kwargs,
                )
        else:
            optimizer = torch.optim.AdamW(
                _get_trainable_params(),
                **optimizer_kwargs,
            )
        
        # One-Cycle LR Scheduler
        lr_cfg = train_cfg.get("lr_schedule", {})
        total_steps = int(lr_cfg.get("total_steps", 5108000))
        max_lr = float(lr_cfg.get("max_lr", 0.0008))
        warmup_pct = float(lr_cfg.get("warmup_pct", 0.05))
        final_div = float(lr_cfg.get("final_div", 10000.0))
        
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
        """Log learning rate before optimizer step.
        
        NOTE: Gradient clipping is handled by PL's gradient_clip_val config.
        Manual clipping here was removed to avoid double-clipping which can
        hurt gradient flow and slow convergence.
        """
        # Log current LR for progress bar (every optimizer step)
        try:
            lr = optimizer.param_groups[0]['lr']
        except (IndexError, KeyError, AttributeError):
            lr = 0.0
        self.log("lr", lr, prog_bar=True, sync_dist=True)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save custom state."""
        # time_elapsed is tracked cumulatively; include resume offset for consistency
        elapsed = time.time() - self._global_start_time + self._resume_time_offset
        self.training_state.time_elapsed = elapsed
        
        checkpoint["training_state"] = {
            "step": self.training_state.step,
            "epoch": self.training_state.epoch,
            "samples_seen": self.training_state.samples_seen,
            "best_val_loss": self.training_state.best_val_loss,
            "train_losses": self.training_state.train_losses,
            "val_losses": self.training_state.val_losses,
            "lr_history": self.training_state.lr_history,
            "lambda_history": self.training_state.lambda_history,
            "time_elapsed": self.training_state.time_elapsed,
            "current_file_idx": self.training_state.current_file_idx,
            "current_position": self.training_state.current_position,
        }
        
        # Save EMA state
        if self.ema_model is not None:
            ema_state = self.ema_model.state_dict()
            # Strip _orig_mod. prefix if EMA model is compiled
            clean_ema_state = {}
            for k, v in ema_state.items():
                clean_key = k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k
                clean_ema_state[clean_key] = v
            checkpoint["ema_model"] = clean_ema_state

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
            # Restore data resume position
            # current_file_idx is a GLOBAL index (includes rank offset from
            # _update_data_position: base_file + rank + files_completed * world_size).
            # Store it as-is so _update_data_position() can add rank back correctly
            # for DDP sharding (rank 0 gets even files, rank 1 gets odd files).
            self.training_state.current_file_idx = ts.get("current_file_idx", 0)
            self.training_state.current_position = ts.get("current_position", 0)
            self._resume_start_file = ts.get("current_file_idx", 0)
            self._resume_start_position = ts.get("current_position", 0)
            # Track previously elapsed time so time budget is correct across resumes
            self._resume_time_offset = ts.get("time_elapsed", 0.0)
            # Track step offset so training_state.step continues from checkpoint value
            # even though PL's trainer.global_step starts at 0 on fresh Trainer
            self._resume_step_offset = ts.get("step", 0)
            # Track samples seen at resume for correct data position estimation
            self._samples_seen_at_resume = ts.get("samples_seen", 0)
        
        # Restore lambda jitter RNG state for exact continuity across resume.
        # NOTE: applied directly here (not deferred to setup()) because PL calls
        # setup() BEFORE on_load_checkpoint(), so deferring would never restore.
        data_state = checkpoint.get("data_state", {})
        _lambda_rng_state = data_state.get("_lambda_rng") if isinstance(data_state, dict) else None
        if _lambda_rng_state is not None:
            try:
                self._lambda_rng.set_state(_lambda_rng_state)
            except Exception:
                # Fallback: reseed per-rank
                try:
                    rank = self.trainer.global_rank
                except Exception:
                    rank = 0
                self._lambda_rng.manual_seed(self.config.get("seed", 42) + rank * 10000)
        
        # Load EMA state (checkpoint_manager.save_checkpoint uses "ema_model" key,
        # PL's on_save_checkpoint uses "ema_state_dict" key - support both)
        ema_state = checkpoint.get("ema_state_dict") or checkpoint.get("ema_model")
        if ema_state and self.ema_model is not None:
            # Strip _orig_mod. prefix for compatibility with compiled/uncompiled models
            clean_ema_state = {}
            for k, v in ema_state.items():
                clean_key = k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k
                clean_ema_state[clean_key] = v
            self.ema_model.load_state_dict(clean_ema_state)

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
        self._elapsed_offset = 0.0  # Previously elapsed time (for resume)

    def on_train_start(self, trainer, pl_module):
        """Adjust start time for previously elapsed time (resume support)."""
        # Account for time already spent training before resume
        training_state = getattr(pl_module, 'training_state', None)
        if training_state is not None:
            self._elapsed_offset = float(getattr(training_state, 'time_elapsed', 0.0))
        self.start_time = time.time()  # Reset to now; elapsed will include offset

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapsed = time.time() - self.start_time + self._elapsed_offset
        if elapsed >= self.max_seconds:
            print(f"Time budget ({self.max_seconds/3600:.1f}h) exceeded. Stopping...")
            trainer.should_stop = True


class OverfittingCallback(Callback):
    """
    Detects overfitting and attempts automatic remediation before giving up.

    Detection (multi-signal):
    1. Val/train loss gap > threshold (relative gap)
    2. Validation loss worsening for N consecutive checks
    3. EMA model outperforming raw model by significant margin

    Remediation (escalating):
    1. Increase weight decay (2x)
    2. Add/increase dropout (0.1 → 0.2 → 0.3)
    3. Reduce learning rate (0.5x)
    4. Increase label smoothing (2x, cap 0.3)
    5. Restore best model + stop training (compute saver)

    If remediation fails to improve val loss after cooldown, moves to next strategy.
    If all strategies exhausted, stops training to avoid wasting compute.
    """

    def __init__(
        self,
        patience: int = 5,
        gap_threshold: float = 0.10,          # relative gap threshold (val-train)/val
        max_remediation_attempts: int = 4,
        remediation_cooldown_checks: int = 3,  # val checks to wait after remediation
        min_improvement: float = 0.002,       # min val loss improvement to count as success
        early_stop_worse_count: int = 15,      # stop if val worse this many checks
    ):
        self.patience = patience
        self.gap_threshold = gap_threshold
        self.max_remediation_attempts = max_remediation_attempts
        self.remediation_cooldown_checks = remediation_cooldown_checks
        self.min_improvement = min_improvement
        self.early_stop_worse_count = early_stop_worse_count

        self.val_losses: List[float] = []
        self.train_losses: List[float] = []
        self.best_val_loss: float = float('inf')
        self.worse_count: int = 0
        self.checks_since_improvement: int = 0
        self.overfitting_detected: bool = False
        self.remediation_attempts: int = 0
        self.checks_since_remediation: int = 0
        self.remediation_history: List[str] = []
        self.best_model_state: Optional[Dict] = None
        self.auto_stop: bool = False
        self.remediation_successful: bool = False
        self._last_val_loss_before_remediation: Optional[float] = None

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only rank 0 makes overfitting decisions in DDP
        if trainer.global_rank != 0:
            return

        metrics = trainer.callback_metrics
        val_loss = metrics.get("val/loss_total")
        train_loss = metrics.get("train/loss_total")

        if val_loss is None or train_loss is None:
            return

        val_loss = float(val_loss)
        train_loss = float(train_loss)
        current_step = pl_module.training_state.step

        self.val_losses.append(val_loss)
        self.train_losses.append(train_loss)
        self.checks_since_remediation += 1

        # Save best model state for potential restore
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.worse_count = 0
            self.checks_since_improvement = 0
            # Strip _orig_mod. prefix from compiled model state_dict keys
            # so EMA model (not compiled) can also load this state
            raw_state = {}
            for k, v in pl_module.model.state_dict().items():
                clean_key = k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k
                raw_state[clean_key] = v.clone()
            self.best_model_state = raw_state
        else:
            self.worse_count += 1
            self.checks_since_improvement += 1

        # Compute relative gap: (val - train) / val
        gap = val_loss - train_loss
        relative_gap = gap / max(val_loss, 1e-8)

        # If we just remediated, wait for cooldown before evaluating
        if self.checks_since_remediation < self.remediation_cooldown_checks:
            # Still check for runaway val loss (emergency stop)
            if self.worse_count >= self.early_stop_worse_count:
                print(f"\n\n🛑 EMERGENCY STOP: Val loss worse for {self.worse_count} checks after remediation")
                print(f"   Best val loss: {self.best_val_loss:.4f}")
                trainer.should_stop = True
                self.auto_stop = True
            return

        # Check if remediation was successful
        if self.overfitting_detected and not self.remediation_successful:
            if self._last_val_loss_before_remediation is not None:
                improvement = self._last_val_loss_before_remediation - val_loss
                if improvement >= self.min_improvement:
                    self.remediation_successful = True
                    print(f"\n✅ Remediation successful: val loss improved by {improvement:.4f}")
                    self.overfitting_detected = False
                    self.worse_count = 0
                    self.checks_since_improvement = 0
                else:
                    print(f"\n⚠️  Remediation did not improve val loss (improvement: {improvement:.4f}, needed: {self.min_improvement})")

        # Check if we should attempt remediation
        should_remediate = (
            relative_gap > self.gap_threshold
            and self.worse_count >= self.patience
            and not self.overfitting_detected
            and self.remediation_attempts < self.max_remediation_attempts
        )

        if should_remediate:
            self._attempt_remediation(pl_module, trainer, gap, val_loss, train_loss)
            return

        # If already overfitting and remediation exhausted, stop
        if self.overfitting_detected and self.remediation_attempts >= self.max_remediation_attempts:
            if not self.remediation_successful:
                # NOTE: worse_count was already incremented on line ~1043 for this
                # same validation check (val_loss >= best_val_loss). Do NOT
                # double-increment here, or early stop triggers twice as fast.
                if self.worse_count >= self.patience:
                    print(f"\n\n🛑 AUTO-STOP: Overfitting not fixable after {self.remediation_attempts} remediation attempts")
                    print(f"   Best val loss: {self.best_val_loss:.4f} (restoring best model)")
                    trainer.should_stop = True
                    self.auto_stop = True
            return

        # Also stop if validation keeps getting worse regardless of gap
        if not self.overfitting_detected and self.worse_count >= self.early_stop_worse_count:
            print(f"\n\n🛑 EARLY STOP: Validation loss worse for {self.worse_count} checks")
            print(f"   Best val loss: {self.best_val_loss:.4f}")
            trainer.should_stop = True

    def _attempt_remediation(self, pl_module, trainer, gap: float, val_loss: float, train_loss: float):
        """Try automatic remediation strategies."""
        self.remediation_attempts += 1
        self.checks_since_remediation = 0
        self.remediation_successful = False
        self._last_val_loss_before_remediation = val_loss
        self.overfitting_detected = True

        strategy = self.remediation_attempts

        print(f"\n\n🔧 REMEDIATION ATTEMPT {strategy}/{self.max_remediation_attempts}")
        rel_gap = gap / max(val_loss, 1e-8)
        print(f"   Relative gap: {rel_gap:.4f} (val={val_loss:.4f}, train={train_loss:.4f})")

        if strategy == 1:
            self._increase_weight_decay(pl_module, factor=2.0)
        elif strategy == 2:
            self._add_dropout(pl_module, p=0.1)
        elif strategy == 3:
            self._reduce_learning_rate(pl_module, factor=0.5)
        elif strategy == 4:
            self._increase_label_smoothing(pl_module, factor=2.0)

        print(f"   Waiting {self.remediation_cooldown_checks} val checks to see if remediation helps...")

    def _increase_weight_decay(self, pl_module, factor: float = 2.0):
        """Increase optimizer weight decay."""
        optimizers = pl_module.optimizers()
        if optimizers is None:
            return
        old_wd = 0.0
        new_wd = 0.0
        modified = False
        for opt in optimizers:
            for group in opt.param_groups:
                old_wd = float(group.get('weight_decay', 0.0))
                new_wd = old_wd * factor
                group['weight_decay'] = new_wd
                print(f"   Weight decay: {old_wd:.2e} → {new_wd:.2e}")
                modified = True
        if modified:
            self.remediation_history.append(f"weight_decay_{old_wd:.2e}_to_{new_wd:.2e}")

    def _add_dropout(self, pl_module, p: float = 0.1):
        """Add dropout to the model (escalating: 0.1 → 0.2 → 0.3)."""
        model = pl_module.model
        # Unwrap compiled model if needed (torch.compile wraps in _orig_mod)
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        if hasattr(model, 'set_dropout'):
            # Escalate dropout on each attempt
            old_p = float(getattr(model, 'dropout_p', 0.0))
            new_p = min(0.5, old_p + p) if old_p > 0 else p
            model.set_dropout(new_p)
            print(f"   Dropout: {old_p:.2f} → {new_p:.2f}")
            self.remediation_history.append(f"dropout_{old_p:.2f}_to_{new_p:.2f}")
        else:
            print("   WARNING: Model doesn't support dynamic dropout")

    def _reduce_learning_rate(self, pl_module, factor: float = 0.5):
        """Reduce learning rate.
        
        Also updates OneCycleLR scheduler's max_lrs so the reduction
        isn't overridden on the next optimizer step.
        """
        optimizers = pl_module.optimizers()
        if optimizers is None:
            return
        old_lr = 0.0
        new_lr = 0.0
        modified = False
        # Capture schedulers once (avoid repeated PL calls inside loop)
        try:
            scheds = pl_module.lr_schedulers()
            scheduler = scheds[0] if scheds else None
            if isinstance(scheduler, dict):
                scheduler = scheduler.get("scheduler")
        except Exception:
            scheduler = None
        for opt in optimizers:
            for i, group in enumerate(opt.param_groups):
                old_lr = float(group['lr'])
                new_lr = old_lr * factor
                group['lr'] = new_lr
                # Also update OneCycleLR max_lrs so reduction sticks
                if scheduler is not None and hasattr(scheduler, 'max_lrs'):
                    try:
                        if i < len(scheduler.max_lrs):
                            scheduler.max_lrs[i] = new_lr
                    except Exception:
                        pass
                print(f"   Learning rate: {old_lr:.2e} → {new_lr:.2e}")
                modified = True
        if modified:
            self.remediation_history.append(f"lr_reduced_{factor}x")

    def _increase_label_smoothing(self, pl_module, factor: float = 2.0):
        """Increase label smoothing on WDL loss."""
        if hasattr(pl_module, 'wdl_loss_fn'):
            old_smoothing = float(pl_module.wdl_loss_fn.label_smoothing)
            new_smoothing = min(0.5, old_smoothing * factor)
            pl_module.wdl_loss_fn.label_smoothing = new_smoothing
            print(f"   Label smoothing: {old_smoothing:.3f} → {new_smoothing:.3f}")
            self.remediation_history.append(f"label_smoothing_{old_smoothing:.3f}_to_{new_smoothing:.3f}")

    def get_learning_status(self) -> Dict[str, Any]:
        """Return detailed learning status including remediation history."""
        if len(self.val_losses) < 2:
            return {"status": "insufficient_data", "learning": True}

        recent_val = self.val_losses[-5:] if len(self.val_losses) >= 5 else self.val_losses
        recent_train = self.train_losses[-5:] if len(self.train_losses) >= 5 else self.train_losses

        val_trend = recent_val[-1] - recent_val[0]
        train_trend = recent_train[-1] - recent_train[0]
        gap = recent_val[-1] - recent_train[-1]
        relative_gap = gap / max(recent_val[-1], 1e-8)

        if self.auto_stop:
            return {"status": "stopped_overfitting", "learning": False, "gap": gap, "relative_gap": relative_gap, "remediation": self.remediation_history}
        elif self.overfitting_detected and self.remediation_attempts < self.max_remediation_attempts:
            return {"status": "remediating", "learning": True, "gap": gap, "relative_gap": relative_gap, "attempt": self.remediation_attempts, "remediation": self.remediation_history}
        elif self.overfitting_detected:
            return {"status": "overfitting_stopped", "learning": False, "gap": gap, "relative_gap": relative_gap, "remediation": self.remediation_history}
        elif val_trend > 0 and train_trend < 0:
            return {"status": "overfitting", "learning": False, "gap": gap, "relative_gap": relative_gap, "remediation": self.remediation_history}
        elif abs(val_trend) < 1e-4:
            return {"status": "plateaued", "learning": False, "gap": gap, "relative_gap": relative_gap}
        else:
            return {"status": "learning", "learning": True, "gap": gap, "relative_gap": relative_gap}

    def restore_best_model(self, pl_module):
        """Restore the best model state (before overfitting)."""
        if self.best_model_state is not None:
            # Unwrap compiled model if needed
            model = pl_module.model
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
            model.load_state_dict(self.best_model_state)
            # EMA model is a standalone NNUEModel (not a submodule), so strip
            # the 'ema_model.' prefix that exists in best_model_state (saved
            # from the full model which has ema_model as a submodule).
            if pl_module.ema_model is not None:
                clean_ema_state = {}
                for k, v in self.best_model_state.items():
                    clean_key = k.replace('ema_model.', '', 1) if k.startswith('ema_model.') else k
                    clean_ema_state[clean_key] = v
                pl_module.ema_model.load_state_dict(clean_ema_state)
            print(f"   ✅ Restored best model (val_loss={self.best_val_loss:.4f})")


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
    # OverfittingCallback detects if model is memorizing vs actually learning
    overfitting_cfg = adv_cfg.get("overfitting", {})
    callbacks = [
        TimeLimitCallback(
            max_hours=time_cfg.get("hours", 9.5),
        ),
        OverfittingCallback(
            patience=overfitting_cfg.get("patience", 5),
            gap_threshold=overfitting_cfg.get("gap_threshold", 0.15),
        ),
        TQDMProgressBar(
            max_steps=max_steps,
            refresh_rate=log_cfg.get("progress_bar_refresh_rate", 10),
        ),
    ]
    
    # Logger
    logger = None
    if log_cfg.get("tensorboard", True):
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
            logger = TensorBoardLogger(
                save_dir=log_cfg.get("dir", "/kaggle/working/logs"),
                name="nnue",
                version=None,
            )
        except ImportError:
            print("  WARNING: TensorBoard not installed, using CSV logger instead")
            logger = None
    
    # Precision
    precision = "16-mixed" if hw_cfg.get("fp16", True) else "32-true"
    
    # Validation interval: val_check_interval can be int (epochs) or float (fraction of epoch)
    # IterableDataset requires limit_val_batches to be int or 1.0 (no float fractions).
    val_check_interval = adv_cfg.get("validation", {}).get("val_check_interval", 0.05)
    if isinstance(val_check_interval, float):
        check_val_every_n_epoch = 1
        # Convert float fraction to int batch count for IterableDataset compatibility
        limit_val_batches = max(1, int(val_check_interval * 100))
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
        accumulate_grad_batches=adv_cfg.get("gradient_accumulation", config.get("training", {}).get("accumulate_grad_batches", 8)),
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