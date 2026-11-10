#!/bin/bash
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

# setup_and_train.sh - One-command setup and training for Kaggle
# Usage: bash setup_and_train.sh [--smoke-test]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Default args
SMOKE_TEST=false
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test)
            SMOKE_TEST=true
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

cd "$PROJECT_ROOT"

log_info "=== NNUE Trainer Setup & Train ==="
log_info "Project root: $PROJECT_ROOT"

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 not found"
    exit 1
fi
log_info "Python: $(python3 --version)"

# Check for Kaggle environment
if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
    log_info "Running on Kaggle: ${KAGGLE_KERNEL_RUN_TYPE}"
    ON_KAGGLE=true
else
    log_info "Running locally"
    ON_KAGGLE=false
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    log_info "GPU info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    log_warn "nvidia-smi not found, assuming CPU-only"
fi

# Install dependencies
log_info "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Compile C++ loader
log_info "Compiling C++ binpack loader..."
if python3 -c "from src.data import compile_cpp_loader; exit(0 if compile_cpp_loader() else 1)"; then
    log_success "C++ loader compiled successfully"
else
    log_warn "C++ loader compilation failed, will use Python fallback"
fi

# Check HF token
if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ "$ON_KAGGLE" == true ]]; then
        log_warn "HF_TOKEN not set in Kaggle Secrets - checkpoint uploads will be disabled"
    else
        log_warn "HF_TOKEN not set in environment - checkpoint uploads will be disabled"
    fi
else
    log_info "HF_TOKEN found, Hugging Face uploads enabled"
fi

# Run training
log_info "Starting training..."
PYTHON_ARGS=("run.py")
if [[ "$SMOKE_TEST" == true ]]; then
    PYTHON_ARGS+=("--smoke-test")
fi
PYTHON_ARGS+=("${EXTRA_ARGS[@]}")

log_info "Command: python3 ${PYTHON_ARGS[*]}"
echo ""

# Execute with proper signal handling
python3 "${PYTHON_ARGS[@]}"
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    log_success "Training completed successfully"
else
    log_error "Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE