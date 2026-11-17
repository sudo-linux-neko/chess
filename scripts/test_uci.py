#!/usr/bin/env python3
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare nnue_core.eval_nnue with a Stockfish 16.1 parity binary.

Build the parity binary with scripts/build_stockfish.sh (its `eval` forces the
Small network). Stockfish prints ``0.01 * to_cp(v)`` with
NormalizeToPawnValue=356, so one printed unit is 3.56 internal units.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nnue_core  # noqa: E402

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "2r3k1/1p3ppp/pq2b3/3p4/3P4/P1N1P3/1P3PPP/2R2RK1 b - - 0 1",
    "4k3/8/8/8/8/8/8/R3K3 b - - 0 1",
    "8/8/8/8/8/8/8/K6k w - - 0 1",
]


def stockfish_values(binary: str, nnue: str, fens: list[str]) -> list[float]:
    cmds = [f"setoption name EvalFileSmall value {nnue}"]
    for fen in fens:
        cmds += [f"position fen {fen}", "eval"]
    cmds.append("quit")
    out = subprocess.run(
        [binary], input="\n".join(cmds) + "\n", text=True, capture_output=True, check=True
    ).stdout
    return [float(v) for v in re.findall(r"NNUE evaluation\s+([+-]?\d+\.\d+)", out)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nnue", required=True)
    p.add_argument("--stockfish", default="/tmp/opencode/sf161-src/src/stockfish-small")
    p.add_argument("--tolerance", type=float, default=4.0, help="internal units")
    args = p.parse_args()

    sf = stockfish_values(args.stockfish, args.nnue, FENS)
    if len(sf) != len(FENS):
        raise SystemExit(f"parsed {len(sf)} evaluations, expected {len(FENS)}")

    bad = 0
    for fen, sf_pawns in zip(FENS, sf):
        feats, _, _, _ = nnue_core.fen_features(fen)
        stm = 0 if fen.split()[1] == "w" else 1
        ours_stm = nnue_core.eval_nnue(
            args.nnue, np.array([feats], dtype=np.int16), np.array([stm], dtype=np.uint8)
        )[0]
        ours_white = ours_stm if stm == 0 else -ours_stm
        sf_internal = sf_pawns * 356.0
        diff = ours_white - sf_internal
        ok = abs(diff) <= args.tolerance
        bad += not ok
        print(
            f"{'ok  ' if ok else 'FAIL'} ours={ours_white:+8.1f} "
            f"sf={sf_internal:+8.1f} diff={diff:+6.1f}  {fen}"
        )
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
