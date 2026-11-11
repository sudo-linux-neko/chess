#!/usr/bin/env python3
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
ULTRA-OPTIMIZED NNUE TRAINER - Single-Cell Kaggle Runner
=========================================================
One-cell solution for Kaggle 2x T4 (16GB each, ~12h session, ~20GB disk)

Usage in Kaggle notebook (single cell):
    !python run.py --config configs/train_ultra.yaml

Or with overrides:
    !python run.py --config configs/train_ultra.yaml --gpus 2 --time-budget 9.5 --lr 0.0008

Features:
- Auto-resume from HF checkpoints across Kaggle sessions
- Streaming data management (only 5GB on disk at a time)
- Predictive background prefetching (zero download wait)
- torch.compile for 1.5-2x speedup
- Mixed precision (fp16) with fused AdamW
- EMA model for evaluation
- Atomic checkpointing every 10 min
- Background HF upload (never blocks training)
- Stockfish-compatible .nnue export with verification
- Graceful Ctrl+C handling (saves checkpoint before exit)
"""

import os
import sys
import time
import signal
import argparse
import shutil
import subprocess
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

import torch
import numpy as np


def check_kaggle_environment():
    """Check if running in Kaggle and print environment info."""
    is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or Path("/kaggle/working").exists()
    
    if is_kaggle:
        print("=" * 60)
        print("KAGGLE ENVIRONMENT DETECTED")
        print("=" * 60)
        print(f"  Working dir: /kaggle/working")
        print(f"  Disk: {shutil_disk_usage('/kaggle/working'):.1f}GB free")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
        else:
            print("  WARNING: No CUDA GPU detected!")
        
        # Check HF_TOKEN
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            print(f"  HF_TOKEN: configured ({len(hf_token)} chars)")
        else:
            print("  HF_TOKEN: NOT SET - add to Kaggle Secrets for HF upload")
        
        print("=" * 60)
    else:
        print("Running outside Kaggle environment")
    
    return is_kaggle


def shutil_disk_usage(path):
    """Get free disk space in GB."""
    import shutil
    stat = shutil.disk_usage(path)
    return stat.free / 1024**3


def install_dependencies():
    """Install required Python packages."""
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        print("Installing dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"pip install warning: {result.stderr[-500:] if result.stderr else 'unknown'}")
        print("Dependencies installed.")


def setup_torch_optimizations():
    """Apply all PyTorch speed optimizations."""
    # TF32 for faster matmul on Ampere+ (T4 supports this)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    # FP16 reduced precision reduction for faster fp16
    if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    
    print("PyTorch optimizations enabled (TF32, cudnn.benchmark, fp16 reduction)")


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-Optimized NNUE Trainer - Single Cell Kaggle Runner"
    )
    parser.add_argument("--config", default="configs/train_ultra.yaml", help="Config file path")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-minute smoke test")
    parser.add_argument("--compile-only", action="store_true", help="Only compile C++ loader")
    parser.add_argument("--build-rust", action="store_true", help="Build Rust loader")
    parser.add_argument("--download-data", action="store_true", help="Only download training data")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark throughput")
    parser.add_argument("--export-only", action="store_true", help="Export latest checkpoint")
    parser.add_argument("--verify-export", action="store_true", help="Verify exported net")
    parser.add_argument("--gpus", type=int, help="Override number of GPUs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--time-budget", type=float, help="Override time budget (hours)")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install")
    parser.add_argument("--skip-clone", action="store_true", help="Skip git clone (already cloned)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ULTRA-OPTIMIZED NNUE TRAINER - Stockfish on 2x T4")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPUs: {torch.cuda.device_count()}")
    
    # Check environment
    is_kaggle = check_kaggle_environment()
    
    # Install dependencies
    if not args.skip_install:
        install_dependencies()
    
    # Setup torch optimizations
    setup_torch_optimizations()
    
    # Load config and run via CLI
    import yaml
    from src.cli import main as cli_main
    
    overrides = {}
    if args.gpus is not None:
        overrides["hardware.gpus"] = args.gpus
    if args.batch_size is not None:
        overrides["training.batch_size"] = args.batch_size
    if args.time_budget is not None:
        overrides["time_budget.hours"] = args.time_budget
    if args.lr is not None:
        overrides["training.learning_rate"] = args.lr
    
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(Path(__file__).parent / config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if overrides:
        for key, value in overrides.items():
            keys = key.split('.')
            cfg = config
            for k in keys[:-1]:
                cfg = cfg.setdefault(k, {})
            cfg[keys[-1]] = value
    
    config["config_path"] = config_path
    
    # Setup directories
    for key in ["data", "checkpoint", "export", "logging"]:
        cfg = config.get(key, {})
        dir_path = cfg.get("dir") or cfg.get("data_dir")
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Handle special modes
    if args.compile_only:
        from src.data import compile_cpp_loader
        success = compile_cpp_loader()
        return 0 if success else 1
    
    if args.build_rust:
        from src.cli import build_rust
        return build_rust(config)
    
    if args.download_data:
        from src.cli import download_data
        return download_data(config)
    
    if args.benchmark:
        from src.cli import run_benchmark
        return run_benchmark(config)
    
    if args.smoke_test:
        from src.cli import run_smoke_test
        return run_smoke_test(config)
    
    if args.export_only:
        from src.cli import run_export_only
        return run_export_only(config)
    
    if args.verify_export:
        from src.cli import run_verify_export
        return run_verify_export(config)
    
    # === MAIN TRAINING MODE ===
    return run_full_training(config, is_kaggle)


def run_full_training(config, is_kaggle):
    """Run full training with all optimizations."""
    from src.data import (
        create_dataloader, compile_cpp_loader, download_training_data,
        StreamingBinpackDataset, DataSource, StreamingDataManager, StreamingConfig,
        StreamingMode
    )
    from src.train import NNUELightningModule, create_trainer
    from src.checkpoint import CheckpointManager, find_resumable_checkpoint
    from src.hf_upload import create_uploader
    from src.export import create_exporter
    from src.model import create_model
    
    print("\n" + "=" * 60)
    print("STARTING FULL TRAINING")
    print("=" * 60)
    
    # 1. Compile C++ binpack loader (non-fatal - Python fallback available)
    print("\n[1/8] Compiling C++ binpack loader...")
    cpp_ok = compile_cpp_loader()
    if cpp_ok:
        print("C++ loader ready (fast path).")
    else:
        print("WARNING: C++ loader compilation failed - using Python fallback (slower but works).")
        print("   To fix: pip install pybind11 cmake")
    
    # 2. Setup data (streaming or full download)
    print("\n[2/8] Preparing training data...")
    data_dir = Path(config["data"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    stream_cfg = config.get("data", {}).get("streaming", {})
    use_streaming = stream_cfg.get("enabled", False)
    
    streaming_manager = None
    if use_streaming:
        print("  Streaming mode ENABLED - predictive prefetching active")
        streaming_config = StreamingConfig(
            mode=StreamingMode(stream_cfg.get("mode", "adaptive")),
            target_gb=stream_cfg.get("target_gb", 5.0),
            min_gb=stream_cfg.get("min_gb", 2.0),
            max_gb=stream_cfg.get("max_gb", 8.0),
            fallback_gb=stream_cfg.get("fallback_gb", 3.0),
            keep_recent=stream_cfg.get("keep_recent", 2),
            auto_cleanup=stream_cfg.get("auto_cleanup", True),
        )
        streaming_manager = StreamingDataManager(data_dir, config, streaming_config)
        stats = streaming_manager.get_stats()
        print(f"  Active: {stats['active_gb']:.1f}GB / {stats['target_gb']}GB target")
        print(f"  Files available: {stats['total_available']}, downloaded: {stats['downloaded']}")
    else:
        binpack_files = list(data_dir.glob("*.binpack"))
        if not binpack_files:
            print("  No data found, downloading from Stockfish sources...")
            download_training_data(data_dir, config, max_files=10)
            binpack_files = list(data_dir.glob("*.binpack"))
        print(f"  Found {len(binpack_files)} data files")
    
    # 3. Initialize components
    print("\n[3/8] Initializing training components...")
    checkpoint_manager = CheckpointManager(config)
    hf_uploader = create_uploader(config)
    
    # Check HF token status
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print(f"  HF_TOKEN: configured ({len(hf_token)} chars) - uploads enabled")
        hf_uploader.start()
    else:
        print("  HF_TOKEN: NOT SET - uploads disabled (training continues without HF)")
        print("   Add HF_TOKEN to Kaggle Secrets for checkpoint uploads")
        # Don't start uploader if no token - FakeHFUploader doesn't need it
    
    # 4. Check for resumable checkpoint
    print("\n[4/8] Checking for resumable checkpoint...")
    resume_ckpt = find_resumable_checkpoint(config, hf_uploader if hf_token else None)
    
    model = create_model(config)
    print(f"  Model parameters: {model.get_num_parameters():,}")
    
    pl_module = NNUELightningModule(config, checkpoint_manager, hf_uploader)
    
    if resume_ckpt:
        print(f"  Resuming from: {resume_ckpt}")
        checkpoint_manager.load_checkpoint(
            resume_ckpt,
            model=pl_module.model,
            optimizer=pl_module.optimizers()[0] if pl_module.optimizers() else None,
            scheduler=pl_module.lr_schedulers()[0] if pl_module.lr_schedulers() else None,
            scaler=None,
            training_state=pl_module.training_state,
            ema_model=pl_module.ema_model,
        )
        # Sync EMA with loaded weights
        if pl_module.ema_model is not None:
            pl_module.ema_model.load_state_dict(pl_module.model.state_dict())
        print(f"  Resumed at step {pl_module.training_state.step}")
    
    # 5. Create trainer
    print("\n[5/8] Creating PyTorch Lightning trainer...")
    trainer = create_trainer(config, checkpoint_manager, hf_uploader)
    
    # 6. Signal handler for graceful shutdown
    print("\n[6/8] Setting up graceful shutdown handler...")
    
    def signal_handler(signum, frame):
        print(f"\n  Received signal {signum}, saving checkpoint immediately...")
        try:
            ckpt_path = checkpoint_manager.save_checkpoint(
                model=pl_module.model,
                optimizer=pl_module.optimizers()[0] if pl_module.optimizers() else None,
                scheduler=pl_module.lr_schedulers()[0] if pl_module.lr_schedulers() else None,
                scaler=trainer.scaler if hasattr(trainer, 'scaler') and trainer.scaler is not None else None,
                training_state=pl_module.training_state,
                config=config,
                ema_model=pl_module.ema_model,
                is_best=True,
            )
            if ckpt_path:
                hf_uploader.queue_upload(ckpt_path)
                meta_path = ckpt_path.with_suffix('.meta.json')
                if meta_path.exists():
                    hf_uploader.queue_upload(meta_path, remote_path=f"checkpoints/{meta_path.name}", priority=5)
            print(f"  Checkpoint saved: {ckpt_path}")
        except Exception as e:
            print(f"  Checkpoint save failed: {e}")
        trainer.should_stop = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 7. Run training
    print("\n[7/8] Starting training loop...")
    print("  Press Ctrl+C for graceful checkpoint save")
    start = time.time()
    
    try:
        trainer.fit(pl_module)
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    except Exception as e:
        print(f"\n  Training error: {e}")
        import traceback
        traceback.print_exc()
        if hf_token:
            try:
                hf_uploader.shutdown(wait=True)
            except Exception:
                pass
        if streaming_manager:
            try:
                streaming_manager.shutdown()
            except Exception:
                pass
        return 1
    
    elapsed = time.time() - start
    print(f"\n  Training completed in {elapsed/3600:.2f} hours")
    print(f"  Total steps: {pl_module.training_state.step}")
    
    # 8. Save final checkpoint, export, and upload
    print("\n[8/8] Finalizing - checkpoint, export, upload...")
    
    ckpt_path = checkpoint_manager.save_checkpoint(
        model=pl_module.model,
        optimizer=pl_module.optimizers()[0] if pl_module.optimizers() else None,
        scheduler=pl_module.lr_schedulers()[0] if pl_module.lr_schedulers() else None,
        scaler=trainer.scaler if hasattr(trainer, 'scaler') and trainer.scaler is not None else None,
        training_state=pl_module.training_state,
        config=config,
        ema_model=pl_module.ema_model,
        is_best=True,
    )
    
    if ckpt_path and hf_token:
        try:
            hf_uploader.queue_upload(ckpt_path)
            meta_path = ckpt_path.with_suffix('.meta.json')
            if meta_path.exists():
                hf_uploader.queue_upload(meta_path, remote_path=f"checkpoints/{meta_path.name}", priority=5)
        except Exception as e:
            print(f"  Checkpoint upload failed: {e}")
    
    # Export final model (EMA)
    print("\n  Exporting final model (.nnue)...")
    exporter = create_exporter(config)
    
    test_loader = create_dataloader(config, is_train=False, streaming_manager=streaming_manager)
    test_batch = next(iter(test_loader))
    test_data = test_batch["features"][:100]
    
    nnue_path = exporter.export(pl_module.model, test_data, use_ema=True)
    
    # Verify export
    if config.get("export", {}).get("verify", True):
        print("  Verifying export...")
        try:
            result = exporter.verify_export(pl_module.model, nnue_path, test_data)
            if not result.get('verified', False):
                print("  WARNING: Export verification had issues")
        except Exception as e:
            print(f"  Verification error: {e}")
    
    if hf_token:
        try:
            hf_uploader.queue_export(nnue_path)
        except Exception as e:
            print(f"  Export upload failed: {e}")
        
        # Wait for uploads to complete
        print("  Waiting for background uploads to complete...")
        try:
            hf_uploader.shutdown(wait=True, timeout=300)
        except Exception as e:
            print(f"  Uploader shutdown warning: {e}")
    else:
        print("  Skipping HF uploads (no HF_TOKEN)")
    
    # Shutdown streaming manager
    if streaming_manager:
        streaming_manager.shutdown()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Final model: {nnue_path}")
    print(f"  Total time: {elapsed/3600:.2f} hours")
    print(f"  Steps: {pl_module.training_state.step}")
    print(f"  HF Repo: {hf_uploader.repo_id if hf_uploader.repo_id else 'N/A'}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    import shutil
    sys.exit(main())