# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

# Set deterministic behavior for tests
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="session", autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    """Get test device (CPU for CI, CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Pytest configuration
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )