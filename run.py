#!/usr/bin/env python3
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Entry point. Examples:

    python run.py train --config configs/train.yaml --kaggle
    python run.py export --checkpoint checkpoints/latest.pt --out final.nnue
    python run.py verify --checkpoint checkpoints/latest.pt --nnue final.nnue
    python run.py info --nnue final.nnue
    python run.py bench --binpack data/tb6dtm.binpack

The Rust core must be installed once:

    pip install maturin
    maturin develop --release -m crates/nnue_core/Cargo.toml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import main

if __name__ == "__main__":
    main()
