# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Hugging Face Uploader for Background Checkpoint Uploads.

Non-blocking, retry with backoff, never crashes training.
"""

import os
import time
import queue
import threading
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future

try:
    from huggingface_hub import HfApi, HfFolder, create_repo, upload_file, list_repo_files
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    HfApi = None
    create_repo = None
    upload_file = None
    list_repo_files = None


@dataclass
class UploadTask:
    """Task for background upload queue."""
    local_path: Path
    repo_path: str
    task_type: str  # "checkpoint", "log", "export"
    retries: int = 0
    max_retries: int = 5
    created_at: float = 0.0


class HFUploader:
    """
    Background uploader for Hugging Face Hub.

    Features:
    - Non-blocking queue-based uploads
    - Exponential backoff retry
    - Configurable concurrency
    - Never crashes training on failure
    - Auto-creates repo if missing
    """

    def __init__(self, config: dict):
        self.config = config
        hf_cfg = config.get("hf", {})

        self.enabled = hf_cfg.get("enabled", True) and HF_HUB_AVAILABLE
        self.repo_id = hf_cfg.get("repo_id", "vaibhav/nnue-trainer-checkpoints")
        self.token_env = hf_cfg.get("token_env", "HF_TOKEN")
        self.private = hf_cfg.get("private", False)
        self.create_repo = hf_cfg.get("create_repo", True)
        self.upload_workers = hf_cfg.get("upload_workers", 2)
        self.max_retries = hf_cfg.get("max_retries", 5)
        self.retry_backoff_base = hf_cfg.get("retry_backoff_base", 1.0)
        self.upload_logs = hf_cfg.get("upload_logs", True)
        self.upload_exports = hf_cfg.get("upload_exports", True)

        self._queue: queue.Queue = queue.Queue()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False
        self._api: Optional[HfApi] = None
        self._token: Optional[str] = None

        if self.enabled:
            self._init_hf()
            self._start_workers()

    def _init_hf(self):
        """Initialize HF API and authentication."""
        # Get token from environment
        self._token = os.environ.get(self.token_env)
        if not self._token:
            print(f"Warning: {self.token_env} not set, HF uploads disabled")
            self.enabled = False
            return

        try:
            self._api = HfApi(token=self._token)

            # Verify token works
            user_info = self._api.whoami()
            print(f"HF authenticated as: {user_info.get('name', 'unknown')}")

            # Create repo if needed
            if self.create_repo:
                self._ensure_repo_exists()

        except Exception as e:
            print(f"Warning: HF initialization failed: {e}")
            self.enabled = False

    def _ensure_repo_exists(self):
        """Create repo if it doesn't exist."""
        try:
            create_repo(
                repo_id=self.repo_id,
                token=self._token,
                private=self.private,
                exist_ok=True,
                repo_type="model",
            )
            print(f"HF repo ensured: {self.repo_id}")
        except Exception as e:
            print(f"Warning: Could not create/verify HF repo: {e}")

    def _start_workers(self):
        """Start background upload workers."""
        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=self.upload_workers)

        # Submit worker tasks
        for i in range(self.upload_workers):
            self._executor.submit(self._worker_loop, i)

        print(f"Started {self.upload_workers} HF upload workers")

    def _worker_loop(self, worker_id: int):
        """Main worker loop."""
        while self._running:
            try:
                task = self._queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                self._process_task(task, worker_id)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")

    def _process_task(self, task: UploadTask, worker_id: int):
        """Process a single upload task with retries."""
        for attempt in range(task.max_retries + 1):
            try:
                print(f"[Worker {worker_id}] Uploading {task.local_path.name} to {self.repo_id}/{task.repo_path} (attempt {attempt + 1})")

                upload_file(
                    path_or_fileobj=str(task.local_path),
                    path_in_repo=task.repo_path,
                    repo_id=self.repo_id,
                    token=self._token,
                    repo_type="model",
                )

                print(f"[Worker {worker_id}] Upload successful: {task.repo_path}")
                return  # Success

            except Exception as e:
                wait_time = self.retry_backoff_base * (2 ** attempt)
                print(f"[Worker {worker_id}] Upload failed (attempt {attempt + 1}): {e}")
                if attempt < task.max_retries:
                    print(f"[Worker {worker_id}] Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[Worker {worker_id}] Upload failed permanently: {task.repo_path}")

    def queue_upload(self, local_path: Path, repo_subdir: str = "checkpoints"):
        """Queue a file for background upload."""
        if not self.enabled:
            return

        if not local_path.exists():
            print(f"Warning: File not found for upload: {local_path}")
            return

        # Determine repo path
        filename = local_path.name
        repo_path = f"{repo_subdir}/{filename}"

        task = UploadTask(
            local_path=local_path,
            repo_path=repo_path,
            task_type="checkpoint",
            max_retries=self.max_retries,
            created_at=time.time(),
        )

        self._queue.put(task)

    def queue_log_upload(self, log_path: Path):
        """Queue a log file for upload."""
        if not self.enabled or not self.upload_logs:
            return
        self.queue_upload(log_path, repo_subdir="logs")

    def queue_export_upload(self, export_path: Path):
        """Queue an exported .nnue file for upload."""
        if not self.enabled or not self.upload_exports:
            return
        self.queue_upload(export_path, repo_subdir="exports")

    def download_latest_checkpoint(self, config: dict) -> Optional[Path]:
        """
        Download the latest checkpoint from HF repo.

        Returns:
            Local path to downloaded checkpoint, or None on failure.
        """
        if not self.enabled or not self._api:
            return None

        try:
            # List files in checkpoints folder
            files = list_repo_files(
                repo_id=self.repo_id,
                token=self._token,
                repo_type="model",
            )

            # Filter checkpoint files
            ckpt_files = [f for f in files if f.startswith("checkpoints/step_") and f.endswith(".ckpt")]
            if not ckpt_files:
                print("No checkpoints found in HF repo")
                return None

            # Sort by step number (descending)
            ckpt_files.sort(key=lambda f: int(f.split("step_")[1].split(".")[0]), reverse=True)
            latest_file = ckpt_files[0]

            # Download
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=latest_file,
                token=self._token,
                repo_type="model",
                local_dir="/kaggle/working/checkpoints",
                local_dir_use_symlinks=False,
            )

            return Path(local_path)

        except Exception as e:
            print(f"Failed to download checkpoint from HF: {e}")
            return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints in HF repo."""
        if not self.enabled or not self._api:
            return []

        try:
            files = list_repo_files(
                repo_id=self.repo_id,
                token=self._token,
                repo_type="model",
            )

            checkpoints = []
            for f in files:
                if f.startswith("checkpoints/step_") and f.endswith(".ckpt"):
                    step = int(f.split("step_")[1].split(".")[0])
                    checkpoints.append({
                        "path": f,
                        "step": step,
                    })

            checkpoints.sort(key=lambda x: x["step"], reverse=True)
            return checkpoints

        except Exception as e:
            print(f"Failed to list HF checkpoints: {e}")
            return []

    def shutdown(self, wait: bool = True, timeout: float = 60.0):
        """Shutdown uploader gracefully."""
        if not self.enabled:
            return

        print("Shutting down HF uploader...")

        # Signal workers to stop
        self._running = False
        for _ in range(self.upload_workers):
            self._queue.put(None)

        if self._executor:
            if wait:
                self._executor.shutdown(wait=True, timeout=timeout)
            else:
                self._executor.shutdown(wait=False)

        # Wait for queue to drain
        if wait:
            try:
                self._queue.join()
            except Exception:
                pass

        print("HF uploader shutdown complete")


class FakeHFUploader:
    """Fake uploader for smoke testing."""

    def __init__(self, config: dict):
        self.config = config
        self.uploaded_files: List[Path] = []

    def queue_upload(self, local_path: Path, repo_subdir: str = "checkpoints"):
        print(f"[FAKE UPLOAD] {local_path} -> {repo_subdir}/{local_path.name}")
        self.uploaded_files.append(local_path)

    def queue_log_upload(self, log_path: Path):
        self.queue_upload(log_path, "logs")

    def queue_export_upload(self, export_path: Path):
        self.queue_upload(export_path, "exports")

    def download_latest_checkpoint(self, config: dict) -> Optional[Path]:
        return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        return []

    def shutdown(self, wait: bool = True, timeout: float = 60.0):
        pass


def create_uploader(config: dict) -> HFUploader:
    """Factory function to create appropriate uploader."""
    if config.get("smoke_test", {}).get("enabled", False) and config.get("smoke_test", {}).get("fake_hf_upload", False):
        return FakeHFUploader(config)
    return HFUploader(config)