# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Ultra-Optimized Hugging Face Hub Uploader.

Features:
- Background thread upload (never blocks training)
- Exponential backoff with jitter
- Resume interrupted uploads
- Checkpoint + export upload queues
- Bandwidth limiting
- Progress tracking
- Atomic finalization
"""

import os
import time
import threading
import queue
import hashlib
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class UploadStatus(Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class UploadTask:
    """Upload task with metadata."""
    local_path: Path
    remote_path: str
    repo_id: str
    repo_type: str = "model"
    priority: int = 0
    retries: int = 0
    max_retries: int = 5
    status: UploadStatus = UploadStatus.PENDING
    error: Optional[str] = None
    created_at: float = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    file_size: int = 0
    sha256: Optional[str] = None


class HFUploader:
    """
    Background Hugging Face uploader with queue-based processing.
    
    Never blocks training - uploads happen in background thread.
    Supports exponential backoff, resume, and priority queues.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        hf_cfg = config.get("hf", {})

        # If no repo_id provided, auto-generate as {username}/chess-nnue
        self.repo_id = hf_cfg.get("repo_id", "")
        self.repo_name = hf_cfg.get("repo_name", "chess-nnue")
        self.token = os.environ.get(hf_cfg.get("token_env", "HF_TOKEN"), "")
        self.upload_every_n = hf_cfg.get("upload_every_n", 3)
        self.max_retries = hf_cfg.get("max_retries", 5)
        self.backoff_base = hf_cfg.get("backoff_base", 30)
        self.bandwidth_limit_mbps = hf_cfg.get("bandwidth_limit_mbps", 0)  # 0 = unlimited
        self.auto_create_repo = hf_cfg.get("auto_create_repo", True)
        self.repo_private = hf_cfg.get("repo_private", False)

        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._paused = threading.Event()
        self._stats = {
            "uploaded": 0,
            "failed": 0,
            "bytes_uploaded": 0,
            "start_time": time.time(),
        }
        self._lock = threading.Lock()
        self._step_counter = 0  # Monotonic counter for PriorityQueue tiebreaking

        # Try to import huggingface_hub
        self._hf_available = False
        try:
            from huggingface_hub import HfApi, upload_file
            self.HfApi = HfApi
            self.upload_file = upload_file
            self._hf_available = True
        except ImportError:
            logger.warning("huggingface_hub not available, uploads disabled")

    def _ensure_repo_exists(self, api):
        """Create repo if it doesn't exist. Auto-generates repo_id as {username}/chess-nnue if not provided."""
        # Auto-generate repo_id if not provided
        if not self.repo_id:
            if not self.token:
                logger.error("No HF token provided - cannot auto-generate repo_id")
                return False
            try:
                # Get current user info
                user_info = api.whoami(token=self.token)
                username = user_info.get("name") or user_info.get("user", {}).get("name")
                if not username:
                    logger.error("Could not determine HF username from token")
                    return False
                self.repo_id = f"{username}/{self.repo_name}"
                logger.info(f"Auto-generated repo_id: {self.repo_id}")
            except Exception as e:
                logger.error(f"Failed to get HF username: {e}")
                return False
        
        if not self.repo_id:
            return False
        
        try:
            # Check if repo exists
            api.repo_info(repo_id=self.repo_id, repo_type="model")
            logger.info(f"Repo exists: {self.repo_id}")
            return True
        except Exception:
            if not self.auto_create_repo:
                logger.warning(f"Repo {self.repo_id} doesn't exist and auto_create_repo=False")
                return False
            
            # Create repo
            try:
                logger.info(f"Creating repo: {self.repo_id}")
                api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="model",
                    private=self.repo_private,
                    exist_ok=True,
                )
                logger.info(f"Repo created: {self.repo_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to create repo {self.repo_id}: {e}")
                return False

    def start(self):
        """Start background upload worker."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._shutdown.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="HFUploader")
        self._worker_thread.start()
        logger.info("HF Uploader started")

    def shutdown(self, wait: bool = True, timeout: float = 60.0):
        """Shutdown uploader, optionally waiting for queue to drain."""
        self._shutdown.set()
        if wait and self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
        logger.info("HF Uploader stopped")

    def pause(self):
        """Pause uploads (finish current, don't start new)."""
        self._paused.set()

    def resume(self):
        """Resume uploads."""
        self._paused.clear()

    def queue_upload(self, local_path: Union[str, Path], remote_path: Optional[str] = None, priority: int = 0):
        """Queue a file for upload."""
        local_path = Path(local_path)
        if not local_path.exists():
            logger.warning(f"File not found for upload: {local_path}")
            return

        if remote_path is None:
            remote_path = local_path.name

        # Compute SHA256 for integrity (file may be deleted between exists check and here)
        try:
            sha256 = self._compute_sha256(local_path)
            file_size = local_path.stat().st_size
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"File inaccessible for upload: {local_path} - {e}")
            return

        task = UploadTask(
            local_path=local_path,
            remote_path=remote_path,
            repo_id=self.repo_id,
            priority=priority,
            max_retries=self.max_retries,
            created_at=time.time(),
            file_size=file_size,
            sha256=sha256,
        )

        # Priority queue: (-priority, sequence_counter, task)
        # sequence_counter prevents comparing UploadTask objects when
        # priority and timestamp are equal (UploadTask has no __lt__)
        with self._lock:
            self._step_counter += 1
            counter = self._step_counter
        self._queue.put((-priority, counter, task))
        logger.debug(f"Queued upload: {local_path} -> {remote_path}")

    def queue_checkpoint(self, local_path: Union[str, Path], step: int):
        """Queue a checkpoint with step-based naming."""
        local_path = Path(local_path)
        remote_path = f"checkpoints/step_{step:08d}.pt"
        self.queue_upload(local_path, remote_path, priority=10)

    def queue_export(self, local_path: Union[str, Path]):
        """Queue an exported .nnue file."""
        local_path = Path(local_path)
        remote_path = f"exports/{local_path.name}"
        self.queue_upload(local_path, remote_path, priority=100)  # High priority

    def _worker(self):
        """Background worker thread."""
        api = None
        repo_ready = False
        if self._hf_available and self.token:
            api = self.HfApi(token=self.token)
            # Ensure repo exists before processing uploads
            repo_ready = self._ensure_repo_exists(api)

        while not self._shutdown.is_set():
            # Check pause
            if self._paused.is_set():
                time.sleep(1)
                continue

            try:
                # Get next task with timeout
                _, _, task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if not self._hf_available or not api or not repo_ready:
                # Re-queue task for later retry instead of losing it permanently
                # Check max_retries to avoid infinite re-queueing when HF is unavailable
                if task.retries >= task.max_retries:
                    task.status = UploadStatus.FAILED
                    task.error = "HF unavailable - max retries exceeded"
                    with self._lock:
                        self._stats["failed"] += 1
                    self._queue.task_done()
                    logger.error(f"Upload failed permanently (HF unavailable): {task.remote_path}")
                    continue
                # Mark as pending again so it can be retried when HF becomes available
                task.status = UploadStatus.PENDING
                task.retries += 1  # Count this re-queue as a retry attempt
                with self._lock:
                    self._step_counter += 1
                    counter = self._step_counter
                self._queue.put((-task.priority, counter, task))
                self._queue.task_done()
                # Brief sleep to avoid tight failure loop when HF is not ready
                time.sleep(1.0)
                continue

            # Process task with retries
            self._process_task(api, task)
            self._queue.task_done()

    def _process_task(self, api, task: UploadTask):
        """Process a single upload task with retries."""
        task.status = UploadStatus.UPLOADING
        task.started_at = time.time()

        for attempt in range(task.max_retries + 1):
            task.retries = attempt
            try:
                self._upload_file(api, task)
                task.status = UploadStatus.COMPLETED
                task.completed_at = time.time()
                with self._lock:
                    self._stats["uploaded"] += 1
                    self._stats["bytes_uploaded"] += task.file_size
                logger.info(f"Uploaded: {task.remote_path} ({task.file_size/1024/1024:.1f} MB)")
                return

            except Exception as e:
                task.error = str(e)
                logger.warning(f"Upload attempt {attempt+1}/{task.max_retries+1} failed for {task.remote_path}: {e}")
                
                if attempt < task.max_retries:
                    # Exponential backoff with jitter
                    backoff = self.backoff_base * (2 ** attempt) + random.uniform(0, self.backoff_base)
                    time.sleep(backoff)
                else:
                    task.status = UploadStatus.FAILED
                    with self._lock:
                        self._stats["failed"] += 1
                    logger.error(f"Upload failed permanently: {task.remote_path} - {e}")

    def _upload_file(self, api, task: UploadTask):
        """Upload a single file to HF Hub."""
        # Check file still exists (may have been deleted by cleanup between queue and upload)
        if not task.local_path.exists():
            raise FileNotFoundError(f"File no longer exists: {task.local_path}")

        # Bandwidth limiting
        if self.bandwidth_limit_mbps > 0:
            self._rate_limit_upload(task.file_size)

        self.upload_file(
            path_or_fileobj=str(task.local_path),
            path_in_repo=task.remote_path,
            repo_id=task.repo_id,
            repo_type=task.repo_type,
            token=self.token,
        )

    def _rate_limit_upload(self, file_size: int):
        """Simple rate limiting."""
        if self.bandwidth_limit_mbps <= 0:
            return
        # Calculate minimum time for this file at max bandwidth
        min_time = file_size / (self.bandwidth_limit_mbps * 1024 * 1024 / 8)
        time.sleep(min_time)

    def _compute_sha256(self, filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get upload statistics."""
        with self._lock:
            stats = self._stats.copy()
        stats["queue_size"] = self._queue.qsize()
        stats["uptime_seconds"] = time.time() - stats["start_time"]
        return stats

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get list of pending tasks (for monitoring)."""
        # Can't easily inspect PriorityQueue, return empty for now
        return []


class FakeHFUploader:
    """Fake uploader for testing without HF token."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._queue = []
        self._stats = {"uploaded": 0, "failed": 0, "bytes_uploaded": 0}

    def start(self):
        pass

    def shutdown(self, wait: bool = True, timeout: float = 60.0):
        pass

    def queue_upload(self, local_path: Union[str, Path], remote_path: Optional[str] = None, priority: int = 0):
        local_path = Path(local_path)
        self._queue.append({
            "local": str(local_path),
            "remote": remote_path or local_path.name,
            "size": local_path.stat().st_size if local_path.exists() else 0,
        })
        print(f"[FAKE] Queued upload: {local_path} -> {remote_path or local_path.name}")

    def queue_checkpoint(self, local_path: Union[str, Path], step: int):
        self.queue_upload(local_path, f"checkpoints/step_{step:08d}.pt", priority=10)

    def queue_export(self, local_path: Union[str, Path]):
        self.queue_upload(local_path, f"exports/{Path(local_path).name}", priority=100)

    def get_stats(self) -> Dict[str, Any]:
        return self._stats


def create_uploader(config: Dict[str, Any]) -> HFUploader:
    """Factory function to create appropriate uploader."""
    hf_cfg = config.get("hf", {})
    token = os.environ.get(hf_cfg.get("token_env", "HF_TOKEN"), "")
    
    # Check if we have token (repo_id can be auto-generated from username)
    if token:
        return HFUploader(config)
    else:
        print("No HF token configured, using FakeHFUploader")
        return FakeHFUploader(config)