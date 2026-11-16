# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Paste into a Kaggle notebook cell (GPU T4 x2, internet on).

It clones this repository if needed, installs the Rust extension, and starts a
2-GPU distributed run with the 9 hour budget from configs/train.yaml.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/thisisforlearn/chess-works.git"
ROOT = Path("/kaggle/working/chess-works")

if not (ROOT / "run.py").exists():
    subprocess.run(["git", "clone", REPO, str(ROOT)], check=True)

os.chdir(ROOT)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "maturin", "huggingface_hub"], check=True)
subprocess.run(
    [sys.executable, "-m", "maturin", "develop", "--release", "-m", "crates/nnue_core/Cargo.toml"],
    check=True,
)

env = dict(os.environ)
token = env.get("HF_TOKEN")
if not token:
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if token:
    from huggingface_hub import login, whoami

    login(token=token, add_to_git_credential=False)
    print("HF account:", whoami()["name"])

subprocess.run(
    [sys.executable, "run.py", "train", "--config", "configs/train.yaml", "--kaggle"],
    check=True,
    env=env,
)
