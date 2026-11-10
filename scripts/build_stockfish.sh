#!/bin/bash
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

# build_stockfish.sh - Build Stockfish compatible with our NNUE architecture
# Usage: bash build_stockfish.sh [STOCKFISH_COMMIT]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/stockfish-build"

# Default to a known compatible commit (post-NNUE merge)
DEFAULT_COMMIT="master"
STOCKFISH_COMMIT="${1:-$DEFAULT_COMMIT}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

cd "$PROJECT_ROOT"

log_info "Building Stockfish at commit: $STOCKFISH_COMMIT"
log_info "Build directory: $BUILD_DIR"

# Check dependencies
for cmd in git make g++; do
    if ! command -v "$cmd" &> /dev/null; then
        log_error "Required command not found: $cmd"
        exit 1
    fi
done

# Clone or update Stockfish
if [[ -d "$BUILD_DIR/.git" ]]; then
    log_info "Updating existing Stockfish repository..."
    cd "$BUILD_DIR"
    git fetch origin
else
    log_info "Cloning Stockfish..."
    git clone https://github.com/official-stockfish/Stockfish.git "$BUILD_DIR"
    cd "$BUILD_DIR"
fi

# Checkout specific commit
log_info "Checking out commit: $STOCKFISH_COMMIT"
git checkout "$STOCKFISH_COMMIT"

# Verify NNUE support
if ! grep -q "NNUE" src/nnue/architecture.h 2>/dev/null; then
    log_warn "NNUE architecture.h not found at this commit"
    log_warn "Make sure you're using a commit with NNUE support (master or nnue-pytorch branch)"
fi

# Build with NNUE enabled (default in modern Stockfish)
log_info "Building Stockfish..."
# Use ARCH=native for local, or specify for Kaggle
ARCH="${ARCH:-x86-64-vnni}"
COMP="${COMP:-gcc}"

make -j"$(nproc)" ARCH="$ARCH" COMP="$COMP" profile-build

# Verify build
STOCKFISH_BIN="$BUILD_DIR/src/stockfish"
if [[ -f "$STOCKFISH_BIN" ]]; then
    log_success "Stockfish built successfully: $STOCKFISH_BIN"
    "$STOCKFISH_BIN" eval --help 2>&1 | head -5
else
    log_error "Build failed - stockfish binary not found"
    exit 1
fi

# Create symlink for convenience
ln -sf "$STOCKFISH_BIN" "$PROJECT_ROOT/stockfish"
log_success "Symlink created: $PROJECT_ROOT/stockfish -> $STOCKFISH_BIN"

echo ""
log_info "Usage with your trained net:"
echo "  ./stockfish"
echo "  UCI"
echo "  setoption name EvalFile value /path/to/your/nn-<hash>.nnue"
echo "  isready"
echo "  go depth 20"