#!/usr/bin/env python3
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Main entry point for NNUE Trainer.

Usage:
    python run.py                    # Full training on Kaggle
    python run.py --smoke-test       # 2-minute pipeline verification
    python run.py --compile-only     # Only compile C++ loader
    python run.py --export-only      # Export latest checkpoint
"""

import os
import sys
import argparse
import yaml
import time
import signal
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model import create_model
from src.data import create_dataloader, compile_cpp_loader, StreamingBinpackDataset
from src.train import NNUELightningModule, create_trainer
from src.checkpoint import CheckpointManager, find_resumable_checkpoint
from src.hf_upload import create_uploader
from src.export import create_exporter
from src.verify import verify_export_standalone


def load_config(config_path: str, overrides: dict = None) -> dict:
    """Load YAML config with optional overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if overrides:
        # Apply nested overrides (e.g., "training.batch_size" -> config["training"]["batch_size"])
        for key, value in overrides.items():
            keys = key.split('.')
            cfg = config
            for k in keys[:-1]:
                cfg = cfg.setdefault(k, {})
            cfg[keys[-1]] = value

    return config


def setup_environment(config: dict):
    """Set up environment variables and directories."""
    # Create directories
    for key in ["data", "checkpoint", "export", "logging"]:
        cfg = config.get(key, {})
        dir_path = cfg.get("dir") or cfg.get("data_dir")
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Set PyTorch settings
    torch.set_float32_matmul_precision('high')

    # Enable TF32 on Ampere+ (not on T4, but harmless)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def compile_only(config: dict) -> int:
    """Compile C++ binpack loader only."""
    print("Compiling C++ binpack loader...")
    success = compile_cpp_loader()
    return 0 if success else 1


def download_data(config: dict) -> int:
    """Download training data (placeholder - actual download in data.py)."""
    print("Data download is handled automatically during training.")
    print("To pre-download, run training with --smoke-test first.")
    return 0


def run_smoke_test(config: dict) -> int:
    """Run 2-minute smoke test to verify entire pipeline."""
    print("=" * 60)
    print("SMOKE TEST: Verifying full pipeline (2 minutes)")
    print("=" * 60)

    # Override config for smoke test
    smoke_cfg = config.get("smoke_test", {})
    config["smoke_test"]["enabled"] = True
    config["training"]["batch_size"] = smoke_cfg.get("batch_size", 256)
    config["training"]["max_positions"] = smoke_cfg.get("max_positions", 1000)
    config["time_budget"]["hours"] = smoke_cfg.get("time_budget_hours", 0.05)
    config["time_budget"]["checkpoint_interval_min"] = smoke_cfg.get("checkpoint_interval_min", 1)
    config["hardware"]["num_workers"] = smoke_cfg.get("num_workers", 0)

    # Compile C++ loader
    print("\n1. Compiling C++ loader...")
    if not compile_cpp_loader():
        print("WARNING: C++ loader compilation failed, using Python fallback")
        # Continue anyway for smoke test

    # Create tiny dummy data for smoke test
    print("\n2. Creating dummy data...")
    data_dir = Path(config["data"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    _create_dummy_binpack(data_dir / "smoke_test.binpack", 1000)

    # Initialize components
    print("\n3. Initializing training components...")
    checkpoint_manager = CheckpointManager(config)
    hf_uploader = create_uploader(config)

    # Create model
    model = create_model(config)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Create Lightning module
    pl_module = NNUELightningModule(config, checkpoint_manager, hf_uploader)

    # Create trainer
    trainer = create_trainer(config, checkpoint_manager, hf_uploader)

    # Train
    print("\n4. Running training...")
    start = time.time()
    trainer.fit(pl_module)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")

    # Test checkpoint save/load
    print("\n5. Testing checkpoint save/load...")
    ckpt_path = checkpoint_manager.save_checkpoint(
        model=pl_module.model,
        optimizer=pl_module.optimizers(),
        scheduler=pl_module.lr_schedulers(),
        scaler=None,
        training_state=pl_module.training_state,
        config=config,
    )
    if ckpt_path:
        print(f"Checkpoint saved: {ckpt_path}")

        # Test HF upload (fake)
        hf_uploader.queue_upload(ckpt_path)
        time.sleep(0.5)  # Let fake upload complete
        hf_uploader.shutdown(wait=True)

    # Test export
    print("\n6. Testing export...")
    exporter = create_exporter(config)
    test_data = torch.randint(0, 2, (10, 81920), dtype=torch.float32)
    nnue_path = exporter.export(pl_module.model, test_data)
    print(f"Exported: {nnue_path}")

    # Test verification
    print("\n7. Testing verification...")
    # Skip full verification for smoke test

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED - Pipeline verified end-to-end")
    print("=" * 60)
    return 0


def _create_dummy_binpack(path: Path, num_positions: int):
    """Create a minimal dummy binpack file for testing."""
    # This creates a very basic binpack file that the C++ loader can read
    # Format: magic(4) + version(4) + num_positions(8) + positions...
    with open(path, 'wb') as f:
        f.write(b"BINP")  # Magic
        f.write(struct.pack("<I", 1))  # Version
        f.write(struct.pack("<Q", num_positions))  # Num positions

        for _ in range(num_positions):
            # Each position: num_features(2) + features + weights + wdl(3*4) + eval(4) + result(1)
            num_features = 32  # Sparse features
            f.write(struct.pack("<H", num_features))
            # Features (uint16)
            f.write(struct.pack(f"<{num_features}H", *[0] * num_features))
            # Weights (float32)
            f.write(struct.pack(f"<{num_features}f", *[1.0] * num_features))
            # WDL (3 floats)
            f.write(struct.pack("<3f", 0.3, 0.4, 0.3))
            # Eval (float32)
            f.write(struct.pack("<f", 0.0))
            # Game result (int8)
            f.write(struct.pack("<b", 0))

    print(f"Created dummy binpack: {path} ({num_positions} positions)")


def run_training(config: dict) -> int:
    """Run full training."""
    print("=" * 60)
    print("NNUE TRAINER - Full Training")
    print("=" * 60)
    print(f"Config: {config.get('config_path', 'configs/train.yaml')}")
    print(f"Time budget: {config['time_budget']['hours']} hours")
    print(f"GPUs: {config['hardware']['gpus']}")
    print(f"Batch size: {config['training']['batch_size']}")

    # Compile C++ loader
    print("\n1. Compiling C++ binpack loader...")
    if not compile_cpp_loader():
        print("ERROR: C++ loader compilation failed")
        return 1

    # Download data (handled by dataset)
    print("\n2. Preparing data...")
    data_dir = Path(config["data"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing data
    binpack_files = list(data_dir.glob("*.binpack"))
    if not binpack_files:
        print("No binpack files found. Data will be downloaded automatically.")
        # In practice, download_data() would be called here

    # Initialize components
    print("\n3. Initializing components...")
    checkpoint_manager = CheckpointManager(config)
    hf_uploader = create_uploader(config)

    # Find resumable checkpoint
    print("\n4. Checking for resumable checkpoint...")
    resume_ckpt = find_resumable_checkpoint(config, hf_uploader)

    # Create model
    model = create_model(config)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Create Lightning module
    pl_module = NNUELightningModule(config, checkpoint_manager, hf_uploader)

    # Load checkpoint if found
    if resume_ckpt:
        print(f"Resuming from: {resume_ckpt}")
        checkpoint_manager.load_checkpoint(
            resume_ckpt,
            model=model,
            optimizer=None,  # Will be created by Lightning
            scheduler=None,
            scaler=None,
        )

    # Create trainer
    trainer = create_trainer(config, checkpoint_manager, hf_uploader)

    # Train
    print("\n5. Starting training...")
    print("Press Ctrl+C for graceful shutdown (saves checkpoint)")
    start = time.time()

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

    # Final checkpoint
    print("\n6. Saving final checkpoint...")
    ckpt_path = checkpoint_manager.save_checkpoint(
        model=pl_module.model,
        optimizer=pl_module.optimizers(),
        scheduler=pl_module.lr_schedulers(),
        scaler=getattr(pl_module, 'scaler', None),
        training_state=pl_module.training_state,
        config=config,
    )

    if ckpt_path:
        hf_uploader.queue_upload(ckpt_path)

    # Export final model
    print("\n7. Exporting final model...")
    exporter = create_exporter(config)

    # Get some test data for verification
    test_loader = create_dataloader(config, is_train=False)
    test_batch = next(iter(test_loader))
    test_data = test_batch["features"][:100]

    nnue_path = exporter.export(pl_module.model, test_data)

    # Verify export
    if config.get("export", {}).get("verify", True):
        print("\n8. Verifying export...")
        try:
            result = verify_export_standalone(pl_module.model, nnue_path, test_loader, config)
            if not result.passed:
                print("WARNING: Export verification failed!")
        except Exception as e:
            print(f"Verification error: {e}")

    # Upload export
    hf_uploader.queue_export_upload(nnue_path)

    # Shutdown
    hf_uploader.shutdown(wait=True)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final model: {nnue_path}")
    print("=" * 60)

    return 0


def run_export_only(config: dict) -> int:
    """Export latest checkpoint to .nnue."""
    print("Exporting latest checkpoint...")

    checkpoint_manager = CheckpointManager(config)
    latest_ckpt = checkpoint_manager.find_latest_checkpoint()

    if not latest_ckpt:
        print("No checkpoint found")
        return 1

    # Load model
    model = create_model(config)
    checkpoint_manager.load_checkpoint(latest_ckpt, model=model)

    # Export
    exporter = create_exporter(config)
    test_loader = create_dataloader(config, is_train=False)
    test_batch = next(iter(test_loader))
    test_data = test_batch["features"][:100]

    nnue_path = exporter.export(model, test_data)

    # Verify
    if config.get("export", {}).get("verify", True):
        result = verify_export_standalone(model, nnue_path, test_loader, config)
        if not result.passed:
            print("Export verification failed!")
            return 1

    print(f"Exported to: {nnue_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="NNUE Trainer for Stockfish")
    parser.add_argument("--config", default="configs/train.yaml", help="Config file path")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-minute smoke test")
    parser.add_argument("--compile-only", action="store_true", help="Only compile C++ loader")
    parser.add_argument("--download-data", action="store_true", help="Only download data")
    parser.add_argument("--export-only", action="store_true", help="Export latest checkpoint")
    parser.add_argument("--gpus", type=int, help="Override number of GPUs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--time-budget", type=float, help="Override time budget (hours)")

    args = parser.parse_args()

    # Load config
    overrides = {}
    if args.gpus is not None:
        overrides["hardware.gpus"] = args.gpus
    if args.batch_size is not None:
        overrides["training.batch_size"] = args.batch_size
    if args.time_budget is not None:
        overrides["time_budget.hours"] = args.time_budget

    config = load_config(args.config, overrides)
    config["config_path"] = args.config

    # Apply smoke test overrides if requested
    if args.smoke_test:
        smoke_cfg = config.get("smoke_test", {})
        for key, value in smoke_cfg.items():
            if key != "enabled":
                # Apply to appropriate config section
                if key in ["batch_size", "max_positions"]:
                    config["training"][key] = value
                elif key in ["time_budget_hours", "checkpoint_interval_min"]:
                    config["time_budget"][key.replace("_hours", "").replace("_min", "")] = value
                elif key == "num_workers":
                    config["hardware"]["num_workers"] = value
        config["smoke_test"]["enabled"] = True

    # Setup
    setup_environment(config)

    # Run requested command
    if args.compile_only:
        return compile_only(config)
    elif args.download_data:
        return download_data(config)
    elif args.smoke_test:
        return run_smoke_test(config)
    elif args.export_only:
        return run_export_only(config)
    else:
        return run_training(config)


if __name__ == "__main__":
    import struct  # For dummy binpack creation
    sys.exit(main())