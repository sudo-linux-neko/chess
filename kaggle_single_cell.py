# ============================================================================
# KAGGLE SINGLE-CELL NNUE TRAINING — Copy-paste this entire block into one cell
# ============================================================================
# Prerequisites (Kaggle):
#   1. Add HF_TOKEN to Kaggle Secrets with write permission
#   2. Set GPU accelerator: 2x T4 (16GB each)
#   3. Set internet: ON
#   4. Session time: ~12 hours (Kaggle free tier max)
# ============================================================================

import os, sys, subprocess, shutil
from pathlib import Path

print("=" * 60)
print("KAGGLE NNUE TRAINER — Single-Cell Deployment")
print("=" * 60)

# -----------------------------------------------------------
# 0. Environment checks
# -----------------------------------------------------------
hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    print("WARNING: HF_TOKEN not found in environment.")
    print("  Add it to Kaggle Settings → Secrets with write permission.")
    print("  Training will continue but checkpoints won't upload to HF.")

print(f"Python: {sys.version}")
print(f"CUDA available: {__import__('torch').cuda.is_available() if 'torch' in sys.modules else 'checking...'}")

# -----------------------------------------------------------
# 1. Clone / update repo
# -----------------------------------------------------------
REPO_URL = "https://github.com/sudo-linux-neko/chess.git"
REPO_DIR = Path("/kaggle/working/chess-nnue")

if not REPO_DIR.exists():
    print("\n[1/8] Cloning repository...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  Clone failed: {result.stderr}")
        print("  Trying full clone...")
        result = subprocess.run(
            ["git", "clone", REPO_URL, str(REPO_DIR)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"  FATAL: Clone failed: {result.stderr}")
            sys.exit(1)
else:
    print("\n[1/8] Repository exists, pulling latest...")
    result = subprocess.run(
        ["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"  Pull failed (may be local changes): {result.stderr}")
        result = subprocess.run(
            ["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", "master"],
            capture_output=True, text=True, timeout=120
        )
        subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", "origin/master"],
                       capture_output=True, text=True, timeout=60)

print(f"  Repo at: {REPO_DIR}")
os.chdir(REPO_DIR)

# -----------------------------------------------------------
# 2. Install dependencies
# -----------------------------------------------------------
print("\n[2/8] Installing dependencies...")
req_file = REPO_DIR / "requirements.txt"
if not req_file.exists():
    req_file = REPO_DIR / "requirements" / "base.txt"

install_cmds = [
    [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
]

if req_file.exists():
    install_cmds.append([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
else:
    # Fallback: install known deps
    install_cmds.append([
        sys.executable, "-m", "pip", "install",
        "torch", "numpy", "pyyaml", "requests", "huggingface_hub",
        "pytorch-lightning", "tqdm", "psutil",
    ])

for cmd in install_cmds:
    print(f"  Running: {' '.join(cmd[:4])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  WARNING: Install step had issues: {result.stderr[-500:]}")
        # Don't fail — continue, may still work

# -----------------------------------------------------------
# 3. Compile C++ binpack loader (non-fatal)
# -----------------------------------------------------------
print("\n[3/8] Compiling C++ binpack loader...")
if (REPO_DIR / "src/build_cpp_loader.py").exists():
    cpp_cmd = [sys.executable, "src/build_cpp_loader.py"]
else:
    cpp_cmd = [sys.executable, "-c", "from src.data import compile_cpp_loader; print(compile_cpp_loader())"]
cpp_result = subprocess.run(
    cpp_cmd,
    capture_output=True, text=True, timeout=300,
    cwd=REPO_DIR
)
if cpp_result.returncode == 0:
    print("  C++ loader ready (fast path).")
else:
    print(f"  C++ compilation failed (non-fatal): {cpp_result.stderr[-300:]}")
    print("  Will use Python fallback (works, slightly slower).")

# -----------------------------------------------------------
# 4. Load / create config
# -----------------------------------------------------------
print("\n[4/8] Loading configuration...")
CONFIG_PATH = REPO_DIR / "configs" / "train_ultra.yaml"
if not CONFIG_PATH.exists():
    # Create minimal working config inline
    config = {
        "hardware": {
            "gpus": 2,
            "fp16": True,
            "precision": "16-mixed",
            "num_workers": 4,
            "prefetch_factor": 2,
            "benchmark_cudnn": True,
        },
        "model": {
            "input_dim": 81920,
            "num_buckets": 64,
            "features_per_bucket": 1280,
            "ft_output": 256,
            "hidden1": 256,
            "hidden2": 32,
            "output_dim": 1,
            "clipped_relu_max": 1.0,
            "dropout": 0.0,
            "init_type": "kaiming_normal",
            "init_gain": 1.0,
        },
        "training": {
            "batch_size": 262144,
            "micro_batch_size": 32768,
            "max_positions": None,
            "learning_rate": 0.0008,
            "weight_decay": 4e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "wdl_lambda": 0.55,
            "label_smoothing": 0.05,
            "score_scale": 370.0,
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
            "lr_schedule": {
                "type": "one_cycle",
                "max_lr": 0.0008,
                "total_steps": 5108000,
                "warmup_pct": 0.05,
                "final_div": 10000.0,
                "anneal_strategy": "cos",
            },
            "lambda_schedule": {
                "type": "cosine_jitter",
                "T0": 5.05e6,
                "delta": -0.3,
                "sample_jitter": 0.0035,
                "batch_jitter": 0.0070,
                "decay": 0.999,
            },
            "curriculum": {
                "enabled": True,
                "early_fen_skip": 18,
                "max_ply": 64,
            },
        },
        "advanced": {
            "gradient_accumulation": 4,
            "ema": {"enabled": True, "decay": 0.9999, "update_every": 1},
            "overfitting": {
                "patience": 5,
                "gap_threshold": 0.15,
                "max_remediation_attempts": 4,
                "remediation_cooldown_checks": 3,
                "min_improvement": 0.002,
                "early_stop_worse_count": 15,
            },
            "validation": {"val_check_interval": 0.05},
        },
        "data": {
            "data_dir": "/kaggle/working/data",
            "max_data_gb": 15.0,
            "shuffle_buffer_mb": 1024,
            "num_files_simultaneous": 6,
            "drop_last_batch": True,
            "sources": [
                {"type": "huggingface", "repo_id": "stockfish/nnue-training", "pattern": "*.binpack", "priority": 1, "max_files": 50},
            ],
            "streaming": {
                "enabled": True,
                "mode": "adaptive",
                "target_gb": 5.0,
                "min_gb": 2.0,
                "max_gb": 8.0,
                "fallback_gb": 3.0,
                "keep_recent": 2,
                "auto_cleanup": True,
            },
        },
        "checkpoint": {
            "dir": "/kaggle/working/checkpoints",
            "keep_last_n": 5,
            "keep_best_n": 3,
            "include_optimizer": True,
            "include_scheduler": True,
            "include_scaler": True,
            "include_rng": True,
            "include_data_position": True,
            "include_atomic_write": True,
        },
        "time_budget": {
            "hours": 11.0,
            "setup_minutes": 10,
            "checkpoint_interval_min": 10,
            "final_save_min": 5,
        },
        "hf": {
            "enabled": True,
            "repo_id": "",  # Auto-generated: {username}/chess-nnue
            "repo_name": "chess-nnue",
            "token_env": "HF_TOKEN",
            "create_repo": True,
            "private": False,
            "upload_workers": 1,
            "max_retries": 5,
            "retry_backoff_base": 30,
            "bandwidth_limit_mbps": 0,
            "upload_logs": False,
            "upload_exports": True,
        },
        "export": {
            "dir": "/kaggle/working/exports",
            "quantize": True,
            "quantization_method": "per_channel_symmetric",
            "verify": True,
            "verify_samples": 20000,
            "verify_tolerance_cp": 0.01,
            "verify_max_error_cp": 0.5,
        },
        "logging": {
            "log_every_n_steps": 10,
            "log_dir": "/kaggle/working/logs",
            "csv_log": True,
            "tensorboard": False,
            "log_gpu_stats": False,
            "progress_bar_refresh_rate": 10,
        },
        "seed": 42,
    }
    # Save config for reproducibility
    import yaml
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  Created default config at {CONFIG_PATH}")
else:
    import yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    print(f"  Loaded config from {CONFIG_PATH}")

# Override HF token from env
config.setdefault("hf", {})["token_env"] = "HF_TOKEN"

# -----------------------------------------------------------
# 5. Check disk space
# -----------------------------------------------------------
print("\n[5/8] Checking disk space...")
disk = shutil.disk_usage("/kaggle/working")
free_gb = disk.free / 1024**3
print(f"  Free disk: {free_gb:.1f} GB")
if free_gb < 5.0:
    print("  WARNING: Less than 5GB free — training may fail due to disk space.")
    print("  Consider reducing streaming target_gb or max_files.")

# -----------------------------------------------------------
# 6. Run training
# -----------------------------------------------------------
print("\n[6/8] Starting training...")
sys.path.insert(0, str(REPO_DIR / "src"))

# Set HF token env for the session
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# Import and run
from run import main

# Build argv for run.py
sys.argv = ["run.py", "--config", str(CONFIG_PATH)]

exit_code = main()

# -----------------------------------------------------------
# 7. Final report
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)
print(f"Exit code: {exit_code}")
print(f"Repo: {REPO_DIR}")
print(f"Checkpoints: {config.get('checkpoint', {}).get('dir', '/kaggle/working/checkpoints')}")
print(f"Exports: {config.get('export', {}).get('dir', '/kaggle/working/exports')}")
if hf_token:
    print(f"HF Repo: auto-generated as {{username}}/chess-nnue")
print("=" * 60)

sys.exit(exit_code)
