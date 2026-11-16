#!/usr/bin/env bash
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Local setup + smoke train. For Kaggle use KAGGLE.md instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -d "$HOME/ml_env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/ml_env/bin/activate"
fi

pip install -r requirements.txt
maturin develop --release -m crates/nnue_core/Cargo.toml
python run.py train --config "${1:-configs/smoke.yaml}"
