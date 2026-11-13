#!/usr/bin/env python3
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
ULTRA-OPTIMIZED NNUE TRAINER - Single-Cell Kaggle Runner
=========================================================
One-cell solution for Kaggle 2x T4 (16GB each, ~12h session, ~20GB disk)

Usage in Kaggle notebook (single cell):
    !python run.py --config configs/train_ultra.yaml

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
- Comprehensive error handling at every step
"""

import os
import sys
import time
import signal
import argparse
import shutil
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))
# Add cpp_loader dir for direct import of compiled C++ loader
CPP_LOADER_DIR = SRC_DIR / "cpp_loader"
if CPP_LOADER_DIR.exists():
    sys.path.insert(0, str(CPP_LOADER_DIR))

import torch
import numpy as np


def shutil_disk_usage(path):
    """Get free disk space in GB."""
    stat = shutil.disk_usage(path)
    return stat.free / 1024**3


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

        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            print(f"  HF_TOKEN: configured ({len(hf_token)} chars)")
        else:
            print("  HF_TOKEN: NOT SET - add to Kaggle Secrets for HF upload")

        print("=" * 60)
    else:
        print("Running outside Kaggle environment")

    return is_kaggle


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

    # Load config
    import yaml
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(Path(__file__).parent / config_path)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Failed to load config {config_path}: {e}")
        return 1

    overrides = {}
    if args.gpus is not None:
        overrides["hardware.gpus"] = args.gpus
    if args.batch_size is not None:
        overrides["training.batch_size"] = args.batch_size
    if args.time_budget is not None:
        overrides["time_budget.hours"] = args.time_budget
    if args.lr is not None:
        overrides["training.learning_rate"] = args.lr

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
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"WARNING: Could not create directory {dir_path}: {e}")

    # Seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable TF32 for faster matmul on T4 (Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

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
    """Run full training with all optimizations and error handling."""

    from src.data import (
        create_dataloader, compile_cpp_loader, download_training_data,
        StreamingDataManager, StreamingConfig,
        StreamingMode
    )
    from src.train import NNUELightningModule, create_trainer, OverfittingCallback
    from src.checkpoint import CheckpointManager, find_resumable_checkpoint
    from src.hf_upload import create_uploader
    from src.export import create_exporter
    from src.model import create_model

    print("\n" + "=" * 60)
    print("STARTING FULL TRAINING")
    print("=" * 60)

    # 1. Compile C++ binpack loader (non-fatal)
    print("\n[1/8] Compiling C++ binpack loader...")
    try:
        cpp_ok = compile_cpp_loader()
        if cpp_ok:
            print("  C++ loader ready (fast path).")
        else:
            print("  WARNING: C++ compilation failed - using Python fallback (works, slightly slower).")
    except Exception as e:
        print(f"  WARNING: C++ compilation error: {e} - using Python fallback.")

    # 2. Setup data with streaming (starts background download, doesn't block)
    print("\n[2/8] Preparing training data (streaming mode)...")
    data_dir = Path(config.get("data", {}).get("data_dir", "/kaggle/working/data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    stream_cfg = config.get("data", {}).get("streaming", {})
    use_streaming = stream_cfg.get("enabled", False)

    streaming_manager = None
    if use_streaming:
        try:
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
            print("  Background downloader started - will prefetch next files automatically")
        except Exception as e:
            print(f"  WARNING: Streaming setup failed: {e}")
            print("  Falling back to non-streaming mode")
            streaming_manager = None

    if streaming_manager is None:
        # Non-streaming: check for existing data
        try:
            binpack_files = list(data_dir.glob("*.binpack"))
            if not binpack_files:
                print("  No data found, downloading initial batch...")
                download_training_data(data_dir, config, max_files=6)
                binpack_files = list(data_dir.glob("*.binpack"))
            print(f"  Found {len(binpack_files)} data files")
        except Exception as e:
            print(f"  ERROR: Data setup failed: {e}")
            return 1

    # 3. Initialize components
    print("\n[3/8] Initializing training components...")
    try:
        checkpoint_manager = CheckpointManager(config)
    except Exception as e:
        print(f"  ERROR: CheckpointManager failed: {e}")
        return 1

    hf_token = os.environ.get("HF_TOKEN", "")
    hf_uploader = None
    try:
        hf_uploader = create_uploader(config)
        if hf_token:
            print(f"  HF_TOKEN: configured ({len(hf_token)} chars) - uploads enabled")
            hf_uploader.start()
        else:
            print("  HF_TOKEN: NOT SET - uploads disabled (training continues without HF)")
            print("   Add HF_TOKEN to Kaggle Secrets for checkpoint uploads")
            # hf_uploader is FakeHFUploader, no need to start
    except Exception as e:
        print(f"  WARNING: HF uploader setup failed: {e} - continuing without uploads")
        hf_uploader = None

    # 4. Check for resumable checkpoint
    print("\n[4/8] Checking for resumable checkpoint...")
    resume_ckpt = None
    try:
        resume_ckpt = find_resumable_checkpoint(config, hf_uploader if hf_token else None)
        if resume_ckpt:
            print(f"  Found resumable checkpoint: {resume_ckpt}")
    except Exception as e:
        print(f"  WARNING: Checkpoint resume check failed: {e} - starting fresh")

    model = None
    pl_module = None
    try:
        model = create_model(config)
        print(f"  Model parameters: {model.get_num_parameters():,}")
    except Exception as e:
        print(f"  ERROR: Model creation failed: {e}")
        return 1

    try:
        pl_module = NNUELightningModule(config, checkpoint_manager, hf_uploader, streaming_manager=streaming_manager)
    except Exception as e:
        print(f"  ERROR: LightningModule creation failed: {e}")
        return 1

    if resume_ckpt:
        try:
            print(f"  Resuming from: {resume_ckpt}")
            # Load model + training state from checkpoint. Do NOT pass manually-created
            # optimizers/schedulers because PL creates new ones in configure_optimizers()
            # during trainer.fit() — state loaded into our manual objects would be lost.
            # Instead, capture state dicts here and restore them in on_train_start().
            checkpoint_result = checkpoint_manager.load_checkpoint(
                resume_ckpt,
                model=pl_module.model,
                scaler=None,
                training_state=pl_module.training_state,
                ema_model=pl_module.ema_model,
            )
            # Store optimizer/scheduler state for restoration in on_train_start()
            # (PL creates its own optimizers during fit(), so we restore state into them there)
            pl_module._optimizer_state = checkpoint_result.get("optimizer_state", {})
            pl_module._scheduler_state = checkpoint_result.get("scheduler_state", {})
            # Store EMA state dict for restoration in on_train_start() after setup() creates the EMA model
            pl_module._ema_state_dict = checkpoint_result.get("ema_state_dict", {})
            # Restore _resume_time_offset so time budget accounts for previously elapsed time
            # (PL's on_load_checkpoint is not called since we load manually before trainer.fit())
            pl_module._resume_time_offset = pl_module.training_state.time_elapsed
            # Track step offset so training_state.step continues from checkpoint value
            # even though PL's trainer.global_step starts at 0 on fresh Trainer
            pl_module._resume_step_offset = pl_module.training_state.step
            # Restore samples_seen_at_resume and resume start position so _update_data_position()
            # correctly estimates post-resume data position (not counting pre-resume samples)
            pl_module._samples_seen_at_resume = pl_module.training_state.samples_seen
            pl_module._resume_start_file = pl_module.training_state.current_file_idx
            pl_module._resume_start_position = pl_module.training_state.current_position
            # Note: EMA model sync is deferred to on_train_start() because setup() hasn't
            # created pl_module.ema_model yet at this point (before trainer.fit()).
            print(f"  Resumed at step {pl_module.training_state.step}")
        except Exception as e:
            print(f"  WARNING: Checkpoint load failed: {e} - starting fresh")

    # 5. Create trainer
    print("\n[5/8] Creating PyTorch Lightning trainer...")
    try:
        trainer = create_trainer(config, checkpoint_manager, hf_uploader)
    except Exception as e:
        print(f"  ERROR: Trainer creation failed: {e}")
        return 1

    # 6. Signal handler for graceful shutdown
    print("\n[6/8] Setting up graceful shutdown handler...")

    def signal_handler(signum, frame):
        # Only rank 0 saves checkpoints in DDP to prevent file corruption
        try:
            if trainer.global_rank != 0:
                trainer.should_stop = True
                return
        except Exception:
            pass
        
        print(f"\n  Received signal {signum}, saving checkpoint immediately...")
        try:
            # Try PL's optimizers first (available after trainer.fit() starts).
            # Fall back to configure_optimizers() if fit() hasn't started yet.
            try:
                optimizers = pl_module.optimizers()
                opt = optimizers[0] if optimizers else None
            except Exception:
                opt = None
            if opt is None:
                # fit() hasn't started yet - create optimizers manually for checkpoint
                try:
                    opt_cfg = pl_module.configure_optimizers()
                    opt = opt_cfg["optimizer"]
                    lr_sched_cfg = opt_cfg.get("lr_scheduler")
                    if isinstance(lr_sched_cfg, dict):
                        sched = lr_sched_cfg["scheduler"]
                    else:
                        sched = lr_sched_cfg
                except Exception:
                    opt = None
                    sched = None
            else:
                try:
                    scheds = pl_module.lr_schedulers()
                    sched = scheds[0] if scheds else None
                except Exception:
                    sched = None
            ckpt_path = checkpoint_manager.save_checkpoint(
                model=pl_module.model,
                optimizer=opt,
                scheduler=sched,
                scaler=trainer.scaler if hasattr(trainer, 'scaler') and trainer.scaler is not None else None,
                training_state=pl_module.training_state,
                config=config,
                ema_model=pl_module.ema_model,
                is_best=False,  # Interrupted model may not be best; don't overwrite best.pt
            )
            if ckpt_path and hf_uploader:
                try:
                    hf_uploader.queue_upload(ckpt_path)
                    meta_path = ckpt_path.with_suffix('.meta.json')
                    if meta_path.exists():
                        hf_uploader.queue_upload(meta_path, remote_path=f"checkpoints/{meta_path.name}", priority=5)
                except Exception as e:
                    print(f"  Checkpoint upload failed: {e}")
            print(f"  Checkpoint saved: {ckpt_path}")
        except Exception as e:
            print(f"  Checkpoint save failed: {e}")
        trainer.should_stop = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 7. Run training
    print("\n[7/8] Starting training loop...")
    print("  Press Ctrl+C for graceful checkpoint save")
    print("  Background downloader will prefetch next files automatically")
    start = time.time()

    try:
        trainer.fit(pl_module)
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        # Save emergency checkpoint on Ctrl+C (signal handler may not always fire)
        try:
            if pl_module and checkpoint_manager:
                print("  Saving emergency checkpoint...")
                try:
                    optimizers = pl_module.optimizers()
                    opt = optimizers[0] if optimizers else None
                except Exception:
                    opt = None
                try:
                    scheds = pl_module.lr_schedulers()
                    sched = scheds[0] if scheds else None
                except Exception:
                    sched = None
                ckpt_path = checkpoint_manager.save_checkpoint(
                    model=pl_module.model,
                    optimizer=opt,
                    scheduler=sched,
                    scaler=trainer.scaler if hasattr(trainer, 'scaler') and trainer.scaler is not None else None,
                    training_state=pl_module.training_state,
                    config=config,
                    ema_model=pl_module.ema_model,
                    is_best=False,
                )
                if ckpt_path and hf_uploader:
                    try:
                        hf_uploader.queue_upload(ckpt_path)
                        meta_path = ckpt_path.with_suffix('.meta.json')
                        if meta_path.exists():
                            hf_uploader.queue_upload(meta_path, remote_path=f"checkpoints/{meta_path.name}", priority=5)
                    except Exception as e:
                        print(f"  Checkpoint upload failed: {e}")
        except Exception as e2:
            print(f"  Emergency checkpoint failed: {e2}")
    except Exception as e:
        print(f"\n  Training error: {e}")
        import traceback
        traceback.print_exc()
        # Try to save emergency checkpoint on crash
        try:
            if pl_module and checkpoint_manager:
                print("  Saving emergency checkpoint...")
                try:
                    optimizers = pl_module.optimizers()
                    opt = optimizers[0] if optimizers else None
                except Exception:
                    opt = None
                try:
                    scheds = pl_module.lr_schedulers()
                    sched = scheds[0] if scheds else None
                except Exception:
                    sched = None
                ckpt_path = checkpoint_manager.save_checkpoint(
                    model=pl_module.model,
                    optimizer=opt,
                    scheduler=sched,
                    scaler=trainer.scaler if hasattr(trainer, 'scaler') and trainer.scaler is not None else None,
                    training_state=pl_module.training_state,
                    config=config,
                    ema_model=pl_module.ema_model,
                    is_best=False,
                )
                if ckpt_path:
                    print("  Emergency checkpoint saved.")
                    if hf_uploader:
                        try:
                            hf_uploader.queue_upload(ckpt_path)
                            meta_path = ckpt_path.with_suffix('.meta.json')
                            if meta_path.exists():
                                hf_uploader.queue_upload(meta_path, remote_path=f"checkpoints/{meta_path.name}", priority=5)
                        except Exception as e:
                            print(f"  Checkpoint upload failed: {e}")
        except Exception as e2:
            print(f"  Emergency checkpoint failed: {e2}")
        if hf_uploader:
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

    # Learning status report + overfitting auto-stop handling
    print("\n" + "=" * 60)
    print("LEARNING STATUS REPORT")
    print("=" * 60)
    overfitting_cb = None
    for cb in trainer.callbacks:
        if isinstance(cb, OverfittingCallback):
            overfitting_cb = cb
            break

    if overfitting_cb:
        status = overfitting_cb.get_learning_status()
        print(f"  Status: {status['status']}")
        print(f"  Learning: {'YES' if status['learning'] else 'NO'}")
        if 'gap' in status:
            print(f"  Train/Val gap: {status['gap']:.4f}")
        if 'relative_gap' in status:
            print(f"  Relative gap: {status['relative_gap']:.4f}")
        print(f"  Best val loss: {overfitting_cb.best_val_loss:.4f}")
        print(f"  Val checks: {len(overfitting_cb.val_losses)}")
        if status.get('remediation'):
            print(f"  Remediation tried: {', '.join(status['remediation'])}")
        if status.get('attempt'):
            print(f"  Remediation attempt: {status['attempt']}/{overfitting_cb.max_remediation_attempts}")

        # If auto-stopped due to overfitting, restore best model immediately
        if status['status'] in ('stopped_overfitting', 'overfitting_stopped') and not status['learning']:
            print(f"\n  🛑 Training stopped early due to overfitting!")
            print(f"  Restoring best model (val_loss={overfitting_cb.best_val_loss:.4f})...")
            overfitting_cb.restore_best_model(pl_module)
        elif status['status'] == 'remediating':
            print(f"\n  ⚠️  Remediation in progress, continuing training...")
    else:
        print("  Overfitting callback not found")

    print(f"  Total steps: {pl_module.training_state.step}")
    print(f"  Total time: {elapsed/3600:.2f} hours")
    print(f"  Best val loss: {pl_module.training_state.best_val_loss:.4f}")
    print(f"  Final train loss: {pl_module.training_state.train_losses[-1] if pl_module.training_state.train_losses else 'N/A'}")
    print(f"  Final val loss: {pl_module.training_state.val_losses[-1] if pl_module.training_state.val_losses else 'N/A'}")
    print("=" * 60)

    # 8. Save final checkpoint, export, and upload
    print("\n[8/8] Finalizing - checkpoint, export, upload...")

    ckpt_path = None
    try:
        # Optimizer/scheduler should be available after trainer.fit()
        try:
            optimizers = pl_module.optimizers()
            opt = optimizers[0] if optimizers else None
        except Exception:
            opt = None
        try:
            scheds = pl_module.lr_schedulers()
            sched = scheds[0] if scheds else None
        except Exception:
            sched = None
        ckpt_path = checkpoint_manager.save_checkpoint(
            model=pl_module.model,
            optimizer=opt,
            scheduler=sched,
            scaler=trainer.scaler if hasattr(trainer, 'scaler') and trainer.scaler is not None else None,
            training_state=pl_module.training_state,
            config=config,
            ema_model=pl_module.ema_model,
            is_best=False,  # Final model may not be the best; don't overwrite best.pt
        )
        if ckpt_path and hf_uploader and hf_token:
            try:
                hf_uploader.queue_upload(ckpt_path)
                meta_path = ckpt_path.with_suffix('.meta.json')
                if meta_path.exists():
                    hf_uploader.queue_upload(meta_path, remote_path=f"checkpoints/{meta_path.name}", priority=5)
            except Exception as e:
                print(f"  Checkpoint upload failed: {e}")
    except Exception as e:
        print(f"  Checkpoint save failed: {e}")

    # Export final model (EMA)
    print("\n  Exporting final model (.nnue)...")
    nnue_path = None
    try:
        exporter = create_exporter(config)
        
        # Try to get test data for export verification
        test_data = None
        try:
            test_loader = create_dataloader(config, is_train=False, streaming_manager=streaming_manager)
            test_batch = next(iter(test_loader))
            test_data = test_batch["features"][:100]
        except Exception as e:
            print(f"  Warning: Could not get test data for export: {e}")
            # Create dummy test data for export calibration
            input_dim = config.get("model", {}).get("input_dim", 81920)
            test_data = torch.zeros(100, input_dim, dtype=torch.float32)

        # Use EMA model if available (EMA typically has better generalization)
        use_ema_export = pl_module.ema_model is not None
        nnue_path = exporter.export(
            pl_module.ema_model if use_ema_export else pl_module.model,
            calibration_data=test_data, use_ema=use_ema_export
        )

        if config.get("export", {}).get("verify", True):
            print("  Verifying export...")
            try:
                # Verify against the same model that was exported
                verify_model = pl_module.ema_model if use_ema_export else pl_module.model
                result = exporter.verify_export(verify_model, nnue_path, test_data)
                if not result.get('verified', False):
                    print("  WARNING: Export verification had issues")
            except Exception as e:
                print(f"  Verification error: {e}")

        # Upload export and shutdown uploader (always, even if export failed)
        if hf_uploader:
            try:
                if hf_token and nnue_path:
                    hf_uploader.queue_export(nnue_path)
                print("  Waiting for background uploads to complete...")
                hf_uploader.shutdown(wait=True, timeout=300)
            except Exception as e:
                print(f"  Uploader shutdown warning: {e}")
        else:
            print("  Skipping HF uploads (no HF_TOKEN)")
    except Exception as e:
        print(f"  Export failed: {e}")
        # Still shutdown uploader on export failure
        if hf_uploader:
            try:
                print("  Waiting for background uploads to complete...")
                hf_uploader.shutdown(wait=True, timeout=300)
            except Exception:
                pass

    # Shutdown streaming manager
    if streaming_manager:
        try:
            streaming_manager.shutdown()
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    if nnue_path:
        print(f"  Final model: {nnue_path}")
    print(f"  Total time: {elapsed/3600:.2f} hours")
    print(f"  Steps: {pl_module.training_state.step}")
    if hf_uploader and getattr(hf_uploader, 'repo_id', None):
        print(f"  HF Repo: {hf_uploader.repo_id}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())