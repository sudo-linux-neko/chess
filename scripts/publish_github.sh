#!/bin/bash
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

# publish_github.sh - Publish the project to GitHub using gh CLI
# Usage: bash scripts/publish_github.sh [REPO_NAME] [--private]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

REPO_NAME="${1:-nnue-trainer}"
PRIVATE=false

# Parse args
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --private)
            PRIVATE=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

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

# Check gh CLI
if ! command -v gh &> /dev/null; then
    log_error "gh CLI not found. Install: https://cli.github.com/"
    exit 1
fi

# Check auth
if ! gh auth status &> /dev/null; then
    log_error "Not authenticated with gh. Run: gh auth login"
    exit 1
fi

# Verify we're in a git repo
if [[ ! -d .git ]]; then
    log_info "Initializing git repository..."
    git init
    git config user.name "Vaibhav"
    git config user.email "vaibhav@example.com"
fi

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    log_warn "Uncommitted changes detected. Committing..."
    git add -A
    git commit -m "Initial commit: NNUE Trainer for Stockfish on Kaggle

Copyright (C) 2026 Vaibhav
License: GPL-3.0-or-later"
fi

# Get current user
GH_USER=$(gh api user --jq .login)
log_info "GitHub user: $GH_USER"

# Check if repo exists
if gh repo view "$GH_USER/$REPO_NAME" &> /dev/null; then
    log_warn "Repository $GH_USER/$REPO_NAME already exists"
    read -p "Push to existing repo? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted"
        exit 0
    fi
else
    # Create repo
    VISIBILITY="--public"
    if [[ "$PRIVATE" == true ]]; then
        VISIBILITY="--private"
    fi

    log_info "Creating repository $GH_USER/$REPO_NAME..."
    gh repo create "$GH_USER/$REPO_NAME" $VISIBILITY \
        --description "Stockfish-style NNUE trainer for Kaggle (2x T4, 10h)" \
        --license "GPL-3.0" \
        --source=. \
        --push
fi

# Set up branch protection (optional)
log_info "Setting up branch protection..."
gh api "repos/$GH_USER/$REPO_NAME/branches/main/protection" \
    --method PUT \
    --field required_status_checks='{"strict":true,"contexts":[]}' \
    --field enforce_admins=true \
    --field required_pull_request_reviews='{"required_approving_review_count":1}' \
    --field restrictions='{}' 2>/dev/null || log_warn "Branch protection setup failed (may need admin)"

# Add topics
log_info "Adding repository topics..."
gh repo edit "$GH_USER/$REPO_NAME" \
    --add-topic "stockfish" \
    --add-topic "nnue" \
    --add-topic "chess" \
    --add-topic "machine-learning" \
    --add-topic "kaggle" \
    --add-topic "pytorch" \
    --add-topic "gpl-3.0"

log_success "Repository published: https://github.com/$GH_USER/$REPO_NAME"
log_info "Next steps:"
echo "  1. Add HF_TOKEN to repository secrets for CI"
echo "  2. Enable GitHub Actions if desired"
echo "  3. Add repository description and README badges"