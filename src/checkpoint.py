# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Checkpoint Management for NNUE Training.

Handles atomic saves, resume logic, local retention, and integration with HF uploader.
"""

import os
import torch
import shutil
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, asdict
from threading import Lock


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint file."""
    step: int
    epoch: int
    wall_time_elapsed: float
    loss: float
    best_val_loss: float
    config_hash: str
    file_size: int
    sha256: str
    timestamp: float


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and retention.

    Features:
    - Atomic writes (temp file + rename)
    - Keep last N checkpoints
    - Resume from local or HF
    - Metadata tracking
    """

    def __init__(self, config: dict):
        self.config = config
        ckpt_cfg = config.get("checkpoint", {})

        self.checkpoint_dir = Path(ckpt_cfg.get("dir", "/kaggle/working/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = ckpt_cfg.get("keep_last_n", 3)
        self.atomic_write = ckpt_cfg.get("atomic_write", True)
        self.include_optimizer = ckpt_cfg.get("include_optimizer", True)
        self.include_scheduler = ckpt_cfg.get("include_scheduler", True)
        self.include_scaler = ckpt_cfg.get("include_scaler", True)
        self.include_rng = ckpt_cfg.get("include_rng", True)
        self.include_data_position = ckpt_cfg.get("include_data_position", True)

        self._lock = Lock()
        self._metadata_file = self.checkpoint_dir / "metadata.json"
        self._metadata: List[CheckpointMetadata] = self._load_metadata()

    def _load_metadata(self) -> List[CheckpointMetadata]:
        """Load checkpoint metadata from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r') as f:
                    data = json.load(f)
                return [CheckpointMetadata(**item) for item in data]
            except Exception:
                pass
        return []

    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        try:
            with open(self._metadata_file, 'w') as f:
                json.dump([asdict(m) for m in self._metadata], f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save metadata: {e}")

    def _compute_sha256(self, filepath: Path) -> str:
        """Compute SHA256 of file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _compute_config_hash(self, config: dict) -> str:
        """Compute hash of relevant config for compatibility checking."""
        # Only hash architecture-affecting config
        relevant_keys = ["model", "training"]
        relevant_config = {k: config.get(k) for k in relevant_keys}
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        scheduler: Union[torch.optim.lr_scheduler._LRScheduler, List, Dict],
        scaler: Optional[torch.cuda.amp.GradScaler],
        training_state: Any,
        config: dict,
    ) -> Optional[Path]:
        """
        Save a full training checkpoint atomically.

        Returns:
            Path to saved checkpoint, or None on failure.
        """
        with self._lock:
            step = getattr(training_state, 'step', 0)
            timestamp = time.time()

            # Prepare checkpoint dict
            checkpoint = {
                "model": model.state_dict(),
                "step": step,
                "epoch": getattr(training_state, 'epoch', 0),
                "wall_time_elapsed": getattr(training_state, 'wall_time_elapsed', 0),
                "best_val_loss": getattr(training_state, 'best_val_loss', float('inf')),
                "config": config,
                "config_hash": self._compute_config_hash(config),
                "timestamp": timestamp,
            }

            # Add optimizer state
            if self.include_optimizer and optimizer is not None:
                if isinstance(optimizer, list):
                    checkpoint["optimizer"] = [opt.state_dict() for opt in optimizer]
                else:
                    checkpoint["optimizer"] = optimizer.state_dict()

            # Add scheduler state
            if self.include_scheduler and scheduler is not None:
                if isinstance(scheduler, list):
                    checkpoint["scheduler"] = [sch.state_dict() for sch in scheduler]
                elif isinstance(scheduler, dict):
                    checkpoint["scheduler"] = {k: v.state_dict() for k, v in scheduler.items()}
                else:
                    checkpoint["scheduler"] = scheduler.state_dict()

            # Add scaler state
            if self.include_scaler and scaler is not None:
                checkpoint["scaler"] = scaler.state_dict()

            # Add RNG states
            if self.include_rng:
                checkpoint["rng_states"] = {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy": __import__('numpy').random.get_state(),
                    "python": __import__('random').getstate(),
                }

            # Add data position
            if self.include_data_position and hasattr(training_state, 'data_position'):
                checkpoint["data_position"] = training_state.data_position

            # Atomic write
            ckpt_filename = f"step_{step:010d}.ckpt"
            tmp_path = self.checkpoint_dir / f"{ckpt_filename}.tmp"
            final_path = self.checkpoint_dir / ckpt_filename

            try:
                torch.save(checkpoint, tmp_path)

                # Verify written file
                if tmp_path.stat().st_size == 0:
                    raise RuntimeError("Checkpoint file is empty")

                # Atomic rename
                if self.atomic_write:
                    tmp_path.rename(final_path)
                else:
                    shutil.move(str(tmp_path), str(final_path))

                # Update metadata
                sha256 = self._compute_sha256(final_path)
                file_size = final_path.stat().st_size

                metadata = CheckpointMetadata(
                    step=step,
                    epoch=getattr(training_state, 'epoch', 0),
                    wall_time_elapsed=getattr(training_state, 'wall_time_elapsed', 0),
                    loss=0.0,  # Will be updated by training loop
                    best_val_loss=getattr(training_state, 'best_val_loss', float('inf')),
                    config_hash=checkpoint["config_hash"],
                    file_size=file_size,
                    sha256=sha256,
                    timestamp=timestamp,
                )

                self._metadata.append(metadata)
                self._prune_old_checkpoints()
                self._save_metadata()

                print(f"Checkpoint saved: {final_path} ({file_size / 1e6:.1f} MB)")
                return final_path

            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                # Clean up temp file
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                return None

    def _prune_old_checkpoints(self):
        """Remove old checkpoints beyond keep_last_n."""
        if len(self._metadata) <= self.keep_last_n:
            return

        # Sort by step
        self._metadata.sort(key=lambda m: m.step)

        # Remove oldest
        to_remove = self._metadata[:-self.keep_last_n]
        for meta in to_remove:
            ckpt_path = self.checkpoint_dir / f"step_{meta.step:010d}.ckpt"
            try:
                ckpt_path.unlink(missing_ok=True)
                print(f"Removed old checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"Warning: Failed to remove {ckpt_path}: {e}")

        self._metadata = self._metadata[-self.keep_last_n:]

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint in local directory."""
        checkpoints = list(self.checkpoint_dir.glob("step_*.ckpt"))
        if not checkpoints:
            return None

        # Sort by step number
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[1]))
        return checkpoints[-1]

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[Union[torch.optim.Optimizer, List]] = None,
        scheduler: Optional[Union[torch.optim.lr_scheduler._LRScheduler, List, Dict]] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore model/optimizer/scheduler/scaler states.

        Returns:
            Dictionary with training_state and other metadata.
        """
        print(f"Loading checkpoint: {checkpoint_path}")

        if map_location is None:
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Load model
        model.load_state_dict(checkpoint["model"])

        # Load optimizer
        if optimizer is not None and "optimizer" in checkpoint:
            opt_state = checkpoint["optimizer"]
            if isinstance(optimizer, list) and isinstance(opt_state, list):
                for opt, state in zip(optimizer, opt_state):
                    opt.load_state_dict(state)
            elif not isinstance(optimizer, list):
                optimizer.load_state_dict(opt_state)

        # Load scheduler
        if scheduler is not None and "scheduler" in checkpoint:
            sch_state = checkpoint["scheduler"]
            if isinstance(scheduler, list) and isinstance(sch_state, list):
                for sch, state in zip(scheduler, sch_state):
                    sch.load_state_dict(state)
            elif isinstance(scheduler, dict) and isinstance(sch_state, dict):
                for k, sch in scheduler.items():
                    if k in sch_state:
                        sch.load_state_dict(sch_state[k])
            elif not isinstance(scheduler, (list, dict)):
                scheduler.load_state_dict(sch_state)

        # Load scaler
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

        # Load RNG states
        if "rng_states" in checkpoint:
            rng = checkpoint["rng_states"]
            torch.set_rng_state(rng.get("torch", torch.get_rng_state()))
            if rng.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            if rng.get("numpy"):
                __import__('numpy').random.set_state(rng["numpy"])
            if rng.get("python"):
                __import__('random').setstate(rng["python"])

        # Return training state for caller to restore
        training_state = {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "wall_time_elapsed": checkpoint.get("wall_time_elapsed", 0),
            "best_val_loss": checkpoint.get("best_val_loss", float('inf')),
            "data_position": checkpoint.get("data_position", {}),
            "rng_states": checkpoint.get("rng_states", {}),
            "config_hash": checkpoint.get("config_hash", ""),
        }

        print(f"Resumed from step {training_state['step']}, epoch {training_state['epoch']}")
        return training_state

    def verify_checkpoint(self, checkpoint_path: Path, config: dict) -> bool:
        """Verify checkpoint integrity and config compatibility."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Check config hash
            saved_hash = checkpoint.get("config_hash", "")
            current_hash = self._compute_config_hash(config)
            if saved_hash != current_hash:
                print(f"Warning: Config hash mismatch! Saved: {saved_hash}, Current: {current_hash}")
                # Don't fail - just warn

            # Check required keys
            required_keys = ["model", "step", "config"]
            for key in required_keys:
                if key not in checkpoint:
                    print(f"Checkpoint missing required key: {key}")
                    return False

            return True

        except Exception as e:
            print(f"Checkpoint verification failed: {e}")
            return False

    def get_all_checkpoints(self) -> List[CheckpointMetadata]:
        """Get list of all local checkpoints sorted by step."""
        return sorted(self._metadata, key=lambda m: m.step)

    def cleanup_all(self):
        """Remove all checkpoints and metadata."""
        for meta in self._metadata:
            ckpt_path = self.checkpoint_dir / f"step_{meta.step:010d}.ckpt"
            ckpt_path.unlink(missing_ok=True)
        self._metadata.clear()
        self._save_metadata()
        self._metadata_file.unlink(missing_ok=True)


def find_resumable_checkpoint(config: dict, hf_uploader: 'HFUploader' = None) -> Optional[Path]:
    """
    Find the latest resumable checkpoint: local first, then HF.

    Args:
        config: Training configuration
        hf_uploader: Optional HF uploader for remote check

    Returns:
        Path to checkpoint, or None if none found.
    """
    ckpt_manager = CheckpointManager(config)

    # 1. Check local
    local_ckpt = ckpt_manager.find_latest_checkpoint()
    if local_ckpt and ckpt_manager.verify_checkpoint(local_ckpt, config):
        print(f"Found local checkpoint: {local_ckpt}")
        return local_ckpt

    # 2. Check HF (if uploader provided)
    if hf_uploader:
        print("No valid local checkpoint, checking Hugging Face...")
        hf_ckpt = hf_uploader.download_latest_checkpoint(config)
        if hf_ckpt:
            # Verify downloaded checkpoint
            if ckpt_manager.verify_checkpoint(hf_ckpt, config):
                print(f"Downloaded and verified HF checkpoint: {hf_ckpt}")
                return hf_ckpt

    return None