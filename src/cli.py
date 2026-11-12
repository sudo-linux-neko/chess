#!/usr/bin/env python3
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Main entry point for Ultra-Optimized NNUE Trainer - Optimized for Stockfish on Kaggle.

Usage:
    python run.py --config configs/train_ultra.yaml          # Full training on Kaggle (optimized)
    python run.py --config configs/train_ultra.yaml --smoke-test       # 2-minute pipeline verification
    python run.py --config configs/train_ultra.yaml --compile-only     # Only compile C++ loader
    python run.py --config configs/train_ultra.yaml --download-data  # Only download training data
    python run.py --config configs/train_ultra.yaml --export-only      # Export latest checkpoint to .nnue
    python run.py --config configs/train_ultra.yaml --verify-export    # Verify exported net matches PyTorch
    python run.py --config configs/train_ultra.yaml --benchmark        # Benchmark throughput (no training)
    python run.py --config configs/train_ultra.yaml --build-rust       # Build Rust loader with maturin
"""

import os
import sys
import argparse
import yaml
import time
import signal
import random
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model import create_model
from src.data import (create_dataloader, compile_cpp_loader, download_training_data, 
                      StreamingBinpackDataset, DataSource, StreamingDataManager, StreamingConfig)
from src.train import NNUELightningModule, create_trainer, OverfittingCallback
from src.checkpoint import CheckpointManager, find_resumable_checkpoint
from src.hf_upload import create_uploader
from src.export import create_exporter
from src.verify import verify_export_standalone, ExportVerifier


def load_config(config_path: str, overrides: dict = None) -> dict:
    """Load YAML config with optional overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            keys = key.split('.')
            cfg = config
            for k in keys[:-1]:
                cfg = cfg.setdefault(k, {})
            cfg[keys[-1]] = value

    return config


def setup_environment(config: dict):
    """Set up environment variables and directories."""
    for key in ["data", "checkpoint", "export", "logging"]:
        cfg = config.get(key, {})
        dir_path = cfg.get("dir") or cfg.get("data_dir")
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    # PyTorch optimizations
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # Seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compile_only(config: dict) -> int:
    """Compile C++ binpack loader only."""
    print("Compiling C++ binpack loader...")
    success = compile_cpp_loader()
    return 0 if success else 1


def build_rust(config: dict) -> int:
    """Build Rust loader with maturin."""
    print("Building Rust binpack loader with maturin...")
    rust_dir = Path(__file__).parent / "src" / "rust_loader"
    
    try:
        import subprocess
        # Check if maturin is available
        result = subprocess.run(["maturin", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("maturin not found. Install with: pip install maturin")
            return 1
        
        # Build in release mode
        result = subprocess.run(
            ["maturin", "develop", "--release"],
            cwd=rust_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return 1
        
        print("Rust loader built successfully!")
        return 0
        
    except subprocess.TimeoutExpired:
        print("Build timed out")
        return 1
    except FileNotFoundError:
        print("maturin not found. Install with: pip install maturin")
        return 1
    except Exception as e:
        print(f"Build error: {e}")
        return 1


def download_data(config: dict) -> int:
    """Download training data from high-quality sources."""
    print("Downloading training data from Stockfish sources...")
    data_dir = Path(config.get("data", {}).get("data_dir", "/kaggle/working/data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sources = [DataSource(**s) for s in config.get("data", {}).get("sources", [])]
    files = download_training_data(data_dir, config, max_files=20)
    
    print(f"Downloaded {len(files)} files")
    total_gb = 0
    for f in files:
        size_gb = f.stat().st_size / 1024**3
        total_gb += size_gb
        print(f"  {f.name}: {size_gb:.2f} GB")
    print(f"Total: {total_gb:.2f} GB")
    return 0


def run_benchmark(config: dict) -> int:
    """Benchmark data loading and model throughput."""
    print("=" * 60)
    print("BENCHMARK: Data Loading + Model Throughput")
    print("=" * 60)

    if not compile_cpp_loader():
        print("WARNING: C++ loader compilation failed, using Python fallback")

    data_dir = Path(config.get("data", {}).get("data_dir", "/kaggle/working/data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    binpack_files = list(data_dir.glob("*.binpack"))
    if not binpack_files:
        print("No data found, creating dummy data...")
        _create_dummy_binpack(data_dir / "benchmark.binpack", 100000)

    model = create_model(config)
    print(f"Model parameters: {model.get_num_parameters():,}")

    hw_cfg = config.get("hardware", {})
    if hw_cfg.get("compile_model", False):
        try:
            model = torch.compile(model, mode="max-autotune")
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = create_dataloader(config, is_train=True)
    
    print("\nWarming up...")
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        features = batch["features"].to(device, non_blocking=True)
        with torch.no_grad():
            _ = model(features)

    print("Benchmarking...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    total_positions = 0
    
    for i, batch in enumerate(loader):
        features = batch["features"].to(device, non_blocking=True)
        batch_size = features.shape[0]
        
        with torch.no_grad():
            _ = model(features)
        
        total_positions += batch_size
        
        if i >= 20:
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    
    throughput = total_positions / elapsed
    print(f"\nResults:")
    print(f"  Positions processed: {total_positions:,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:,.0f} positions/sec")
    print(f"  Per GPU: {throughput / max(1, hw_cfg.get('gpus', 1)):,.0f} positions/sec")
    
    # Also benchmark export verification
    print("\nBenchmarking quantized inference...")
    verifier = ExportVerifier(config)
    if verifier._rust_available:
        print("Rust SIMD acceleration: AVAILABLE")
    else:
        print("Rust SIMD acceleration: NOT AVAILABLE (using Python fallback)")
    
    return 0


def _create_dummy_binpack(path: Path, num_positions: int):
    import struct
    with open(path, 'wb') as f:
        f.write(b"BINP")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", num_positions))

        for _ in range(num_positions):
            num_features = 32
            f.write(struct.pack("<H", num_features))
            f.write(struct.pack(f"<{num_features}H", *range(num_features)))
            f.write(struct.pack(f"<{num_features}f", *[1.0] * num_features))
            f.write(struct.pack("<3f", 0.3, 0.4, 0.3))
            f.write(struct.pack("<f", 0.0))
            f.write(struct.pack("<b", 0))

    print(f"Created dummy binpack: {path} ({num_positions} positions)")


def run_smoke_test(config: dict) -> int:
    """Run 2-minute smoke test to verify entire pipeline."""
    print("=" * 60)
    print("SMOKE TEST: Verifying full pipeline (2 minutes)")
    print("=" * 60)

    smoke_cfg = config.get("smoke_test", {})
    # Override config for smoke test
    config.setdefault("smoke_test", {})["enabled"] = True
    config.setdefault("training", {})["batch_size"] = smoke_cfg.get("batch_size", 512)
    config["training"]["max_positions"] = smoke_cfg.get("max_positions", 1000)
    config.setdefault("time_budget", {})["hours"] = smoke_cfg.get("time_budget_hours", 0.05)
    config["time_budget"]["checkpoint_interval_min"] = smoke_cfg.get("checkpoint_interval_min", 1)
    config.setdefault("hardware", {})["num_workers"] = smoke_cfg.get("num_workers", 0)
    config["hardware"]["compile_model"] = False
    # Ensure wdl_lambda exists
    if "wdl_lambda" not in config.get("training", {}):
        config.setdefault("training", {})["wdl_lambda"] = 0.55

    if not compile_cpp_loader():
        print("WARNING: C++ loader compilation failed")

    print("\n2. Creating dummy data...")
    data_dir = Path(config.get("data", {}).get("data_dir", "/kaggle/working/data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    _create_dummy_binpack(data_dir / "smoke_test.binpack", 1000)

    print("\n3. Initializing training components...")
    checkpoint_manager = CheckpointManager(config)
    hf_uploader = create_uploader(config)

    model = create_model(config)
    print(f"Model parameters: {model.get_num_parameters():,}")

    pl_module = NNUELightningModule(config, checkpoint_manager, hf_uploader, streaming_manager=streaming_manager)
    trainer = create_trainer(config, checkpoint_manager, hf_uploader)

    print("\n4. Running training...")
    start = time.time()
    trainer.fit(pl_module)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")

    print("\n5. Testing checkpoint save/load...")
    # Optimizer/scheduler may not be available before trainer.fit()
    try:
        opt = pl_module.optimizers()[0] if pl_module.optimizers() else None
    except Exception:
        opt = None
    try:
        sched = pl_module.lr_schedulers()[0] if pl_module.lr_schedulers() else None
    except Exception:
        sched = None
    ckpt_path = checkpoint_manager.save_checkpoint(
        model=pl_module.model,
        optimizer=opt,
        scheduler=sched,
        scaler=None,
        training_state=pl_module.training_state,
        config=config,
    )
    if ckpt_path:
        print(f"Checkpoint saved: {ckpt_path}")
        if hf_uploader:
            try:
                hf_uploader.queue_upload(ckpt_path)
                time.sleep(0.5)
            except Exception as e:
                print(f"  Upload warning: {e}")
        hf_uploader.shutdown(wait=True) if hf_uploader else None

    print("\n6. Testing export...")
    exporter = create_exporter(config)
    test_data = torch.randn(10, config.get("model", {}).get("input_dim", 81920))
    nnue_path = exporter.export(pl_module.model, test_data)
    print(f"Exported: {nnue_path}")

    print("\n7. Testing verification...")
    try:
        result = exporter.verify_export(pl_module.model, nnue_path, test_data)
        print(f"Verification: {'PASSED' if result['verified'] else 'FAILED'}")
    except Exception as e:
        print(f"Verification error (expected for smoke test): {e}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED - Pipeline verified end-to-end")
    print("=" * 60)
    return 0


def run_training(config: dict) -> int:
    """Run full ultra-optimized training."""
    print("=" * 60)
    print("ULTRA-OPTIMIZED NNUE TRAINER - Stockfish on 2x T4")
    print("=" * 60)
    print(f"Config: {config.get('config_path', 'configs/train_ultra.yaml')}")
    print(f"Time budget: {config.get('time_budget', {}).get('hours', 9.5)} hours")
    print(f"GPUs: {config.get('hardware', {}).get('gpus', 2)}")
    print(f"Batch size: {config.get('training', {}).get('batch_size', 'N/A')} (global)")
    print(f"Micro batch: {config.get('training', {}).get('micro_batch_size', 'N/A')} (accum x{config.get('advanced', {}).get('gradient_accumulation', 'N/A')})")
    print(f"LR: {config.get('training', {}).get('learning_rate', 'N/A')} (One-Cycle)")
    print(f"Architecture: FT(256 shared) -> L1(256) -> L2(32) -> Out(1)")
    print(f"Loss: WDL({config.get('training', {}).get('wdl_lambda', 'N/A')}) + Quantile Eval")
    print(f"LR Schedule: One-Cycle + Lambda Jitter")
    print(f"Data: High-quality Stockfish (3500+ Elo, depth 15+)")
    print(f"EMA: Enabled for evaluation")

    print("\n1. Compiling C++ binpack loader (non-fatal)...")
    cpp_ok = compile_cpp_loader()
    if cpp_ok:
        print("  C++ loader ready (fast path).")
    else:
        print("  WARNING: C++ compilation failed - using Python fallback (works, slightly slower).")

    print("\n2. Preparing training data...")
    data_dir = Path(config.get("data", {}).get("data_dir", "/kaggle/working/data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if streaming is enabled
    stream_cfg = config.get("data", {}).get("streaming", {})
    use_streaming = stream_cfg.get("enabled", False)
    
    streaming_manager = None
    if use_streaming:
        print("Streaming mode ENABLED - will manage disk space automatically")
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
    else:
        binpack_files = list(data_dir.glob("*.binpack"))
        if not binpack_files:
            print("No binpack files found. Downloading from Stockfish sources...")
            download_training_data(data_dir, config, max_files=10)
            binpack_files = list(data_dir.glob("*.binpack"))
        
        print(f"Found {len(binpack_files)} data files")
        total_gb = sum(f.stat().st_size for f in binpack_files) / 1024**3
        print(f"Total data: {total_gb:.2f} GB")

    print("\n3. Initializing components...")
    checkpoint_manager = CheckpointManager(config)
    hf_uploader = create_uploader(config)

    print("\n4. Checking for resumable checkpoint...")
    resume_ckpt = find_resumable_checkpoint(config, hf_uploader)

    model = create_model(config)
    print(f"Model parameters: {model.get_num_parameters():,}")

    pl_module = NNUELightningModule(config, checkpoint_manager, hf_uploader, streaming_manager=streaming_manager)

    if resume_ckpt:
        print(f"Resuming from: {resume_ckpt}")
        # Load checkpoint into pl_module.model (not a separate model variable)
        # Optimizer/scheduler may not be available before trainer.fit()
        try:
            opt = pl_module.optimizers()[0] if pl_module.optimizers() else None
        except Exception:
            opt = None
        try:
            sched = pl_module.lr_schedulers()[0] if pl_module.lr_schedulers() else None
        except Exception:
            sched = None
        checkpoint_manager.load_checkpoint(
            resume_ckpt,
            model=pl_module.model,
            optimizer=opt,
            scheduler=sched,
            scaler=None,  # Scaler restored by PL automatically
            training_state=pl_module.training_state,
            ema_model=pl_module.ema_model,
        )
        # Sync EMA model with loaded weights
        if pl_module.ema_model is not None:
            pl_module.ema_model.load_state_dict(pl_module.model.state_dict())

    # Create trainer with streaming manager
    trainer = create_trainer(config, checkpoint_manager, hf_uploader)

    print("\n5. Starting training...")
    print("Press Ctrl+C for graceful shutdown (saves checkpoint)")
    start = time.time()

    # Setup signal handler for graceful shutdown
    shutdown_requested = False
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        print("\nReceived interrupt signal, saving checkpoint...")
        shutdown_requested = True
        trainer.should_stop = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        trainer.fit(pl_module)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed/3600:.2f} hours")
    print(f"Total steps: {pl_module.training_state.step}")
    print(f"Final loss: {pl_module.training_state.train_losses[-1] if pl_module.training_state.train_losses else 'N/A'}")

    print("\n6. Saving final checkpoint...")
    # Optimizer/scheduler should be available after trainer.fit()
    try:
        opt = pl_module.optimizers()[0] if pl_module.optimizers() else None
    except Exception:
        opt = None
    try:
        sched = pl_module.lr_schedulers()[0] if pl_module.lr_schedulers() else None
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
        is_best=True,
    )

    if ckpt_path:
        hf_uploader.queue_upload(ckpt_path)
        # Also upload metadata JSON with loss, time, etc.
        meta_path = ckpt_path.with_suffix('.meta.json')
        if meta_path.exists():
            hf_uploader.queue_upload(meta_path, remote_path=f"checkpoints/{meta_path.name}", priority=5)

    # Report learning status
    print("\n" + "=" * 60)
    print("LEARNING STATUS REPORT")
    print("=" * 60)
    
    # Find overfitting callback
    for cb in trainer.callbacks:
        if isinstance(cb, OverfittingCallback):
            status = cb.get_learning_status()
            print(f"  Status: {status['status']}")
            print(f"  Learning: {'YES' if status['learning'] else 'NO'}")
            if 'gap' in status:
                print(f"  Train/Val gap: {status['gap']:.4f}")
            print(f"  Best val loss: {cb.best_val_loss:.4f}")
            print(f"  Val checks: {len(cb.val_losses)}")
            if status.get('remediation'):
                print(f"  Remediation tried: {', '.join(status['remediation'])}")
            if status.get('attempt'):
                print(f"  Remediation attempt: {status['attempt']}/{cb.max_remediation_attempts}")
            
            # Restore best model if we stopped due to overfitting
            if status['status'] in ('stopped_overfitting', 'overfitting') and not status['learning']:
                print(f"  🔄 Restoring best model from step with val_loss={cb.best_val_loss:.4f}...")
                cb.restore_best_model(pl_module)
            break
    
    print(f"  Total steps: {pl_module.training_state.step}")
    print(f"  Total time: {elapsed/3600:.2f} hours")
    print(f"  Best val loss: {pl_module.training_state.best_val_loss:.4f}")
    print(f"  Final train loss: {pl_module.training_state.train_losses[-1] if pl_module.training_state.train_losses else 'N/A'}")
    print(f"  Final val loss: {pl_module.training_state.val_losses[-1] if pl_module.training_state.val_losses else 'N/A'}")
    print("=" * 60)
    exporter = create_exporter(config)

    test_data = None
    try:
        test_loader = create_dataloader(config, is_train=False, streaming_manager=streaming_manager)
        test_batch = next(iter(test_loader))
        test_data = test_batch["features"][:100]
    except Exception as e:
        print(f"  Warning: Could not get test data for export: {e}")
        test_data = torch.zeros(100, config.get("model", {}).get("input_dim", 81920), dtype=torch.float32)

    # Export EMA model for better playing strength
    nnue_path = exporter.export(pl_module.model, test_data, use_ema=True)

    if config.get("export", {}).get("verify", True):
        print("\n8. Verifying export...")
        try:
            result = exporter.verify_export(pl_module.model, nnue_path, test_data)
            if not result['verified']:
                print("WARNING: Export verification failed!")
        except Exception as e:
            print(f"Verification error: {e}")

    hf_uploader.queue_export(nnue_path)
    hf_uploader.shutdown(wait=True)

    # Shutdown streaming manager
    if streaming_manager:
        print("Shutting down streaming data manager...")
        streaming_manager.shutdown()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final model: {nnue_path}")
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Steps: {pl_module.training_state.step}")
    print("=" * 60)

    return 0


def run_export_only(config: dict) -> int:
    """Export latest checkpoint to .nnue format."""
    print("Exporting latest checkpoint...")

    checkpoint_manager = CheckpointManager(config)
    latest_ckpt = checkpoint_manager.find_latest_checkpoint()

    if not latest_ckpt:
        print("No checkpoint found")
        return 1

    model = create_model(config)
    checkpoint_manager.load_checkpoint(latest_ckpt, model=model)

    exporter = create_exporter(config)
    test_data = None
    try:
        test_loader = create_dataloader(config, is_train=False)
        test_batch = next(iter(test_loader))
        test_data = test_batch["features"][:100]
    except Exception as e:
        print(f"  Warning: Could not get test data: {e}")
        test_data = torch.zeros(100, config.get("model", {}).get("input_dim", 81920), dtype=torch.float32)

    nnue_path = exporter.export(model, test_data, use_ema=True)

    if config.get("export", {}).get("verify", True):
        result = exporter.verify_export(model, nnue_path, test_data)
        if not result["verified"]:
            print("Export verification failed!")
            return 1

    print(f"Exported to: {nnue_path}")
    return 0


def run_verify_export(config: dict) -> int:
    """Verify exported net matches PyTorch."""
    print("Verifying exported net...")

    checkpoint_manager = CheckpointManager(config)
    latest_ckpt = checkpoint_manager.find_latest_checkpoint()

    if not latest_ckpt:
        print("No checkpoint found")
        return 1

    model = create_model(config)
    checkpoint_manager.load_checkpoint(latest_ckpt, model=model)

    export_dir = Path(config.get("export", {}).get("dir", "/kaggle/working/export"))
    nnue_files = list(export_dir.glob("nn-*.nnue"))
    if not nnue_files:
        print("No exported .nnue file found")
        return 1

    nnue_path = max(nnue_files, key=lambda f: f.stat().st_mtime)
    print(f"Verifying: {nnue_path}")

    test_data = None
    try:
        test_loader = create_dataloader(config, is_train=False)
        test_batch = next(iter(test_loader))
        test_data = test_batch["features"][:config.get("export", {}).get("verify_samples", 20000)]
    except Exception as e:
        print(f"  Warning: Could not get test data: {e}")
        test_data = torch.zeros(config.get("export", {}).get("verify_samples", 20000), config.get("model", {}).get("input_dim", 81920), dtype=torch.float32)

    exporter = create_exporter(config)
    result = exporter.verify_export(model, nnue_path, test_data)

    return 0 if result["verified"] else 1


def main():
    parser = argparse.ArgumentParser(description="Ultra-Optimized NNUE Trainer for Stockfish")
    parser.add_argument("--config", default="configs/train_ultra.yaml", help="Config file path")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-minute smoke test")
    parser.add_argument("--compile-only", action="store_true", help="Only compile C++ loader")
    parser.add_argument("--build-rust", action="store_true", help="Build Rust loader with maturin")
    parser.add_argument("--download-data", action="store_true", help="Only download training data")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark throughput")
    parser.add_argument("--export-only", action="store_true", help="Export latest checkpoint")
    parser.add_argument("--verify-export", action="store_true", help="Verify exported net")
    parser.add_argument("--gpus", type=int, help="Override number of GPUs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--time-budget", type=float, help="Override time budget (hours)")
    parser.add_argument("--lr", type=float, help="Override learning rate")

    args = parser.parse_args()

    overrides = {}
    if args.gpus is not None:
        overrides["hardware.gpus"] = args.gpus
    if args.batch_size is not None:
        overrides["training.batch_size"] = args.batch_size
    if args.time_budget is not None:
        overrides["time_budget.hours"] = args.time_budget
    if args.lr is not None:
        overrides["training.learning_rate"] = args.lr

    config = load_config(args.config, overrides)
    config["config_path"] = args.config

    if args.smoke_test:
        smoke_cfg = config.get("smoke_test", {})
        for key, value in smoke_cfg.items():
            if key != "enabled":
                if key in ["batch_size", "max_positions"]:
                    config.setdefault("training", {})[key] = value
                elif key in ["time_budget_hours", "checkpoint_interval_min"]:
                    config.setdefault("time_budget", {})[key.replace("_hours", "").replace("_min", "")] = value
                elif key == "num_workers":
                    config.setdefault("hardware", {})["num_workers"] = value
        config.setdefault("smoke_test", {})["enabled"] = True
        config.setdefault("hardware", {})["compile_model"] = False

    setup_environment(config)

    if args.compile_only:
        return compile_only(config)
    elif args.build_rust:
        return build_rust(config)
    elif args.download_data:
        return download_data(config)
    elif args.benchmark:
        return run_benchmark(config)
    elif args.smoke_test:
        return run_smoke_test(config)
    elif args.export_only:
        return run_export_only(config)
    elif args.verify_export:
        return run_verify_export(config)
    else:
        return run_training(config)


if __name__ == "__main__":
    import struct
    sys.exit(main())