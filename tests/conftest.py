# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
nnue_core = pytest.importorskip("nnue_core")


@pytest.fixture(scope="session")
def sample_binpack() -> str:
    for p in (
        ROOT / "data" / "tb6dtm.binpack",
        Path("/tmp/opencode/data/tb6dtm.binpack"),
    ):
        if p.exists() and p.stat().st_size > 0:
            return str(p)
    pytest.skip("no sample binpack available")
