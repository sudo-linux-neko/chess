#!/usr/bin/env bash
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Point origin at the fork, commit as Vaibhav, and (only with --push) push.
# Never commits checkpoints, data, or nets (see .gitignore).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FORK="${FORK:-https://github.com/thisisforlearn/chess-works.git}"
UPSTREAM="${UPSTREAM:-https://github.com/sudo-linux-neko/chess.git}"

git remote remove origin 2>/dev/null || true
git remote remove upstream 2>/dev/null || true
git remote add origin "$FORK"
git remote add upstream "$UPSTREAM"

git config user.name "Vaibhav"
git config user.email "thisisforlearn@users.noreply.github.com"

git status --short
git add -A
if ! git diff --cached --quiet; then
    git commit -m "Rewrite trainer: real binpacks, HalfKAv2_hm, exact SF16.1 export

- Rust nnue_core: threaded binpack stream (~10 Mpos/s), exact features,
  exact .nnue writer and integer inference
- PyTorch: SF16.1 Small model, win-rate loss, DDP, time-budgeted run,
  atomic exact-resume checkpoints, optional HF upload
- Verified against Stockfish 16.1 (EvalFileSmall) on real positions"
fi

if [ "${1:-}" = "--push" ]; then
    git push -u origin master
else
    echo "not pushed; run: $0 --push"
fi
