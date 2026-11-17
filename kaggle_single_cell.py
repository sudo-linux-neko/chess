# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Paste into one Kaggle notebook cell (GPU T4 x2, Internet on).

Prerequisite: Kaggle Add-ons -> Secrets -> add `HF_TOKEN` with a write token.
This clones/updates the repo, installs Rust if missing, builds the extension,
logs in to Hugging Face, trains for the 9h budget on both T4s, uploads
checkpoints, and prints the final net info.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/thisisforlearn/chess-works.git"
ROOT = Path("/kaggle/working/chess-works")
FRESH = True  # True = ignore local/HF checkpoints; set False only to resume


def run(cmd, **kwargs):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


if shutil.which("cargo") is None:
    run(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        shell=True,
    )
os.environ["PATH"] = f"{Path.home() / '.cargo' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"

if (ROOT / ".git").exists():
    run(["git", "-C", str(ROOT), "fetch", "origin"])
    run(["git", "-C", str(ROOT), "reset", "--hard", "origin/master"])
else:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    run(["git", "clone", REPO, str(ROOT)])
os.chdir(ROOT)
if FRESH:
    shutil.rmtree(ROOT / "checkpoints", ignore_errors=True)
free_gib = shutil.disk_usage(ROOT.parent).free / 2**30
print(f"Free disk: {free_gib:.1f} GiB", flush=True)

run([sys.executable, "-m", "pip", "install", "-q", "maturin>=1.7", "huggingface_hub>=0.23", "pyyaml>=6"])
run([sys.executable, "-m", "maturin", "build", "--release", "-m", "crates/nnue_core/Cargo.toml"])
wheels = sorted((ROOT / "crates/nnue_core/target/wheels").glob("nnue_core-*.whl"))
if not wheels:
    raise SystemExit("maturin produced no wheel; check the build log above")
run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", str(wheels[-1])])

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami

try:
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:
    raise SystemExit("Add a Kaggle secret named HF_TOKEN (write token), then rerun.") from exc
os.environ["HF_TOKEN"] = token
login(token=token, add_to_git_credential=False)
print("HF account:", whoami()["name"], flush=True)

train_env = os.environ.copy()
if FRESH:
    # Ignore latest.pt on the Hub too, not just the local checkpoints directory.
    train_env["NNUE_NO_RESUME"] = "1"
run(
    [sys.executable, "run.py", "train", "--config", "configs/train.yaml", "--kaggle"],
    env=train_env,
)
net = ROOT / "checkpoints" / "final.nnue"
if net.exists():
    run([sys.executable, "run.py", "info", "--nnue", str(net)])
else:
    print("No final.nnue (training did not reach export).", flush=True)
