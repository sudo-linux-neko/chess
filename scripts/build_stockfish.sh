#!/usr/bin/env bash
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build Stockfish 16.1 plus a parity binary whose `eval` uses the Small net
# (the stock build hardcodes Big in Eval::trace).
set -euo pipefail

SRC="${1:-/tmp/opencode/sf161-src}"
OUT="${2:-/tmp/opencode/sf161-src/src}"
JOBS="${JOBS:-$(nproc)}"

if [ ! -d "$SRC" ]; then
    git clone --depth 1 --branch sf_16.1 \
        https://github.com/official-stockfish/Stockfish.git "$SRC"
fi

cd "$SRC/src"
make -j"$JOBS" build ARCH=x86-64-avx2
cp stockfish stockfish-small
sed -i 's/v = NNUE::evaluate<NNUE::Big>(pos, false);/v = NNUE::evaluate<NNUE::Small>(pos, false);/' evaluate.cpp
make -j"$JOBS" build ARCH=x86-64-avx2
cp stockfish "$OUT/stockfish-small"

echo "stockfish:       $SRC/src/stockfish"
echo "stockfish-small: $OUT/stockfish-small"
