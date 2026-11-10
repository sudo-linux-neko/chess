#!/usr/bin/env python3
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Main entry point - delegates to src.cli.main()
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())