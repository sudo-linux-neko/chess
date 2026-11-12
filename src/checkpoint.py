# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Ultra-Optimized Checkpoint Management.

Features:
- Atomic writes with .tmp rename pattern
- Full state restoration (model, optimizer, scheduler, scaler, RNG, data position)
- Keep-last-N + keep-best-N rotation
- Metadata tracking (step, loss, time, git sha)
- Background upload queue integration
- Corruption detection with SHA256
"""

import os
import json
import time
import hashlib
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    step: int
    epoch: int
    samples_seen: int
    train_loss: float
    val_loss: float
    best_val_loss: float
    learning_rate: float
    lambda_val: float
    time_elapsed: float
    timestamp: str
    git_sha: Optional[str] = None
    config_hash: Optional[str] = None
    file_sha256: Optional[str] = None


class CheckpointManager:
    """
    Manages training checkpoints with atomic writes and rotation.
    
    Features:
    - Atomic writes: write to .tmp, fsync, rename
    - Keep-last-N: always keep N most recent
    - Keep-best-N: keep N best by validation loss
    - Metadata sidecar: .meta.json for quick inspection
    - Corruption detection: SHA256 verification
    - Exact resume: full state including RNG and data position
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ckpt_cfg = config.get("checkpoint", {})

        self.checkpoint_dir = Path(ckpt_cfg.get("dir", "/kaggle/working/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = ckpt_cfg.get("keep_last_n", 5)
        self.keep_best_n = ckpt_cfg.get("keep_best_n", 3)
        self.save_optimizer = ckpt_cfg.get("save_optimizer", True)
        self.save_scheduler = ckpt_cfg.get("save_scheduler", True)
        self.save_scaler = ckpt_cfg.get("save_scaler", True)
        self.save_rng = ckpt_cfg.get("save_rng", True)
        self.save_data_state = ckpt_cfg.get("save_data_state", True)
        self.atomic_writes = ckpt_cfg.get("atomic_writes", True)

        self._lock = threading.Lock()
        self._metadata_cache: Dict[str, CheckpointMetadata] = {}

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        scheduler: Optional[LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        training_state: Any,
        config: Dict[str, Any],
        data_state: Optional[Dict[str, Any]] = None,
        ema_model: Optional[nn.Module] = None,
        is_best: bool = False,
        suffix: str = "",
    ) -> Optional[Path]:
        """Save complete training checkpoint atomically."""
        with self._lock:
            step = getattr(training_state, 'step', 0)
            timestamp = datetime.utcnow().isoformat() + 'Z'

            # Prepare state dict
            state = {
                "model": model.state_dict(),
                "training_state": self._serialize_training_state(training_state),
                "config": config,
                "timestamp": timestamp,
                "step": step,
            }

            # Strip _orig_mod. prefix from torch.compile models for compatibility
            # when loading into non-compiled models (and vice versa)
            if "model" in state:
                clean_model_state = {}
                for k, v in state["model"].items():
                    clean_key = k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k
                    clean_model_state[clean_key] = v
                state["model"] = clean_model_state

            # Optional components
            if self.save_optimizer and optimizer is not None:
                state["optimizer"] = optimizer.state_dict()
            if self.save_scheduler and scheduler is not None:
                state["scheduler"] = scheduler.state_dict()
            if self.save_scaler and scaler is not None:
                state["scaler"] = scaler.state_dict()
            if self.save_rng:
                state["rng"] = self._get_rng_state()
            if self.save_data_state and data_state is not None:
                state["data_state"] = data_state
            if ema_model is not None:
                state["ema_model"] = ema_model.state_dict()

            # Generate filename
            if is_best:
                filename = f"best{suffix}.pt"
            else:
                filename = f"step_{step:08d}{suffix}.pt"

            tmp_path = self.checkpoint_dir / f"{filename}.tmp"
            final_path = self.checkpoint_dir / filename

            try:
                # Atomic write
                if self.atomic_writes:
                    self._atomic_save(state, tmp_path, final_path)
                else:
                    torch.save(state, final_path)

                # Compute SHA256
                file_sha256 = self._compute_sha256(final_path)

                # Create metadata
                metadata = CheckpointMetadata(
                    step=step,
                    epoch=getattr(training_state, 'epoch', 0),
                    samples_seen=getattr(training_state, 'samples_seen', 0),
                    train_loss=getattr(training_state, 'train_losses', [0])[-1] if getattr(training_state, 'train_losses', None) else 0,
                    val_loss=getattr(training_state, 'val_losses', [0])[-1] if getattr(training_state, 'val_losses', None) else 0,
                    best_val_loss=getattr(training_state, 'best_val_loss', float('inf')),
                    learning_rate=optimizer.param_groups[0]['lr'] if optimizer else 0,
                    lambda_val=getattr(training_state, 'lambda_history', [0])[-1] if getattr(training_state, 'lambda_history', None) else 0,
                    time_elapsed=getattr(training_state, 'time_elapsed', 0),
                    timestamp=timestamp,
                    git_sha=self._get_git_sha(),
                    config_hash=self._hash_config(config),
                    file_sha256=file_sha256,
                )

                # Save metadata sidecar
                meta_path = final_path.with_suffix('.meta.json')
                with open(meta_path, 'w') as f:
                    json.dump(asdict(metadata), f, indent=2)

                # Update cache
                self._metadata_cache[str(final_path)] = metadata

                # Rotation
                self._rotate_checkpoints(is_best)

                print(f"Checkpoint saved: {final_path} (step={step}, sha256={file_sha256[:16]}...)")
                return final_path

            except Exception as e:
                print(f"Checkpoint save failed: {e}")
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                return None

    def _atomic_save(self, state: Dict, tmp_path: Path, final_path: Path):
        """Save to temp file, fsync, then atomic rename."""
        torch.save(state, tmp_path)
        # Ensure data is on disk
        with open(tmp_path, 'rb') as f:
            f.flush()
            os.fsync(f.fileno())
        # Atomic rename
        tmp_path.rename(final_path)

    def _serialize_training_state(self, training_state: Any) -> Dict:
        """Serialize training state to dict."""
        if hasattr(training_state, '__dict__'):
            return {k: v for k, v in training_state.__dict__.items() 
                   if not k.startswith('_')}
        return dict(training_state) if isinstance(training_state, dict) else {}

    def _get_rng_state(self) -> Dict:
        """Get all RNG states for exact reproducibility."""
        return {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": __import__('numpy').random.get_state(),
            "python": __import__('random').getstate(),
        }

    def _set_rng_state(self, state: Dict):
        """Restore all RNG states."""
        torch.set_rng_state(state["torch"])
        if state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        __import__('numpy').random.set_state(state["numpy"])
        __import__('random').setstate(state["python"])

    def _compute_sha256(self, filepath: Path) -> str:
        """Compute SHA256 of file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _verify_sha256(self, filepath: Path, expected: str) -> bool:
        """Verify file integrity."""
        return self._compute_sha256(filepath) == expected

    def _get_git_sha(self) -> Optional[str]:
        """Get current git commit SHA."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _hash_config(self, config: Dict) -> str:
        """Hash config for reproducibility tracking."""
        import yaml
        config_str = yaml.dump(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _rotate_checkpoints(self, is_best: bool = False):
        """Rotate checkpoints: keep last N + best N."""
        # Get all checkpoint files with robust step parsing
        def _step_from_path(p: Path) -> int:
            try:
                return int(p.stem.split('_')[1])
            except (IndexError, ValueError):
                return -1
        
        checkpoints = sorted(
            [p for p in self.checkpoint_dir.glob("step_*.pt") if _step_from_path(p) >= 0],
            key=_step_from_path,
        )
        best_checkpoints = sorted(
            self.checkpoint_dir.glob("best*.pt"),
            key=lambda p: self._get_val_loss(p),
        )

        # Remove old step checkpoints
        while len(checkpoints) > self.keep_last_n:
            old = checkpoints.pop(0)
            old.unlink(missing_ok=True)
            old.with_suffix('.meta.json').unlink(missing_ok=True)

        # Remove old best checkpoints (keep only best N)
        while len(best_checkpoints) > self.keep_best_n:
            old = best_checkpoints.pop(-1)  # Remove worst
            old.unlink(missing_ok=True)
            old.with_suffix('.meta.json').unlink(missing_ok=True)

    def _get_val_loss(self, path: Path) -> float:
        """Get validation loss from metadata."""
        meta_path = path.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f).get('val_loss', float('inf'))
        return float('inf')

    def load_checkpoint(
        self,
        path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        training_state: Optional[Any] = None,
        data_state: Optional[Dict] = None,
        ema_model: Optional[nn.Module] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint with full state restoration."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Verify integrity
        meta_path = path.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            if not self._verify_sha256(path, metadata.get('file_sha256', '')):
                raise RuntimeError(f"Checkpoint corruption detected: {path}")

        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location='cpu')

        # Load model - strip _orig_mod. prefix for compatibility with compiled/uncompiled models
        model_state = checkpoint["model"]
        clean_model_state = {}
        for k, v in model_state.items():
            clean_key = k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k
            clean_model_state[clean_key] = v
        model.load_state_dict(clean_model_state, strict=strict)

        # Load optimizer
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scheduler
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

        # Load scaler
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

        # Load RNG
        if "rng" in checkpoint:
            self._set_rng_state(checkpoint["rng"])

        # Load EMA
        if ema_model is not None and "ema_model" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_model"])

        # Return training state and data state for manual restoration
        result = {
            "training_state": checkpoint.get("training_state", {}),
            "data_state": checkpoint.get("data_state", {}),
            "config": checkpoint.get("config", {}),
            "step": checkpoint.get("step", 0),
        }

        if training_state is not None and "training_state" in checkpoint:
            self._restore_training_state(training_state, checkpoint["training_state"])

        if data_state is not None and "data_state" in checkpoint:
            data_state.update(checkpoint["data_state"])

        print(f"Checkpoint loaded: step={result['step']}")
        return result

    def _restore_training_state(self, training_state: Any, state_dict: Dict):
        """Restore training state from dict."""
        for key, value in state_dict.items():
            if hasattr(training_state, key):
                setattr(training_state, key, value)

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find most recent checkpoint by step number."""
        checkpoints = list(self.checkpoint_dir.glob("step_*.pt"))
        if not checkpoints:
            return None
        def _step_from_path(p: Path) -> int:
            try:
                return int(p.stem.split('_')[1])
            except (IndexError, ValueError):
                return -1
        valid = [p for p in checkpoints if _step_from_path(p) >= 0]
        if not valid:
            return None
        return max(valid, key=_step_from_path)

    def find_best_checkpoint(self) -> Optional[Path]:
        """Find best checkpoint by validation loss."""
        checkpoints = list(self.checkpoint_dir.glob("best*.pt"))
        if not checkpoints:
            return None
        return min(checkpoints, key=self._get_val_loss)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints with metadata."""
        result = []
        for ckpt in sorted(self.checkpoint_dir.glob("*.pt")):
            meta_path = ckpt.with_suffix('.meta.json')
            if meta_path.exists():
                with open(meta_path) as f:
                    result.append(json.load(f))
            else:
                result.append({
                    "path": str(ckpt),
                    "step": int(ckpt.stem.split('_')[1]) if '_' in ckpt.stem else 0,
                    "size_mb": ckpt.stat().st_size / 1024**2,
                })
        return result


def find_resumable_checkpoint(config: Dict, hf_uploader: Any) -> Optional[Path]:
    """Find the best checkpoint to resume from (local or HF)."""
    checkpoint_manager = CheckpointManager(config)
    
    # Try local first
    local_ckpt = checkpoint_manager.find_latest_checkpoint()
    if local_ckpt and local_ckpt.exists():
        return local_ckpt

    # Try HF Hub
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
        hf_cfg = config.get("hf", {})
        repo_id = hf_cfg.get("repo_id", "")
        # If no repo_id, try to construct from username (same logic as HFUploader)
        if not repo_id:
            token = os.environ.get(hf_cfg.get("token_env", "HF_TOKEN"), "")
            if token:
                from huggingface_hub import HfApi
                api = HfApi(token=token)
                user_info = api.whoami(token=token)
                username = user_info.get("name") or user_info.get("user", {}).get("name")
                if username:
                    repo_id = f"{username}/{hf_cfg.get('repo_name', 'chess-nnue')}"
        
        if repo_id:
            files = list_repo_files(repo_id=repo_id, repo_type="model")
            ckpt_files = [f for f in files if f.endswith(".pt") and "step_" in f]
            if ckpt_files:
                # Get latest by step number - robust parsing
                def _step_from_filename(fname: str) -> int:
                    try:
                        # Expected: step_NNNNNNNN.pt
                        base = fname.split('_')[1].split('.')[0]
                        return int(base)
                    except (IndexError, ValueError):
                        return -1
                
                latest = max(ckpt_files, key=_step_from_filename)
                if _step_from_filename(latest) < 0:
                    print(f"WARNING: Could not parse step from {latest}, skipping HF resume")
                    return None
                ckpt_dir = config.get("checkpoint", {}).get("dir", "/kaggle/working/checkpoints")
                local_path = Path(ckpt_dir) / latest
                print(f"Downloading checkpoint from HF: {latest}")
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=latest,
                        repo_type="model",
                        local_dir=ckpt_dir,
                    )
                except Exception as e:
                    print(f"HF checkpoint download failed: {e}")
                    return None
                return local_path
    except Exception as e:
        print(f"HF checkpoint download failed: {e}")

    return None