#!/usr/bin/env python3
# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Test UCI integration with exported .nnue net.

Usage:
    python scripts/test_uci.py --net /path/to/nn-<hash>.nnue [--stockfish /path/to/stockfish]
"""

import argparse
import subprocess
import tempfile
import os
import sys
from pathlib import Path
import chess
import chess.engine


def test_stockfish_eval(net_path: Path, stockfish_path: Path = Path("stockfish")) -> bool:
    """Test Stockfish EvalFile with the exported net."""
    print(f"Testing Stockfish with net: {net_path}")
    print(f"Stockfish binary: {stockfish_path}")

    if not stockfish_path.exists():
        print(f"ERROR: Stockfish not found at {stockfish_path}")
        print("Run: bash scripts/build_stockfish.sh")
        return False

    if not net_path.exists():
        print(f"ERROR: Net file not found: {net_path}")
        return False

    # Test 1: Basic UCI handshake
    print("\n1. Testing UCI handshake...")
    try:
        proc = subprocess.run(
            [str(stockfish_path)],
            input="uci\nisready\nquit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "id name Stockfish" in proc.stdout and "readyok" in proc.stdout:
            print("   ✓ UCI handshake successful")
        else:
            print(f"   ✗ UCI handshake failed: {proc.stdout}")
            return False
    except Exception as e:
        print(f"   ✗ UCI test error: {e}")
        return False

    # Test 2: Load net via EvalFile
    print("\n2. Testing EvalFile option...")
    try:
        proc = subprocess.run(
            [str(stockfish_path)],
            input=f"uci\nsetoption name EvalFile value {net_path}\nisready\nquit\n",
            capture_output=True,
            text=True,
            timeout=15,
        )
        if "readyok" in proc.stdout:
            print("   ✓ Net loaded successfully")
        else:
            print(f"   ✗ Failed to load net: {proc.stdout}")
            return False
    except Exception as e:
        print(f"   ✗ EvalFile test error: {e}")
        return False

    # Test 3: Basic evaluation
    print("\n3. Testing evaluation...")
    try:
        # Create a simple FEN test
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        proc = subprocess.run(
            [str(stockfish_path)],
            input=f"uci\nsetoption name EvalFile value {net_path}\nisready\nposition fen {fen}\neval\nquit\n",
            capture_output=True,
            text=True,
            timeout=15,
        )
        if "Total evaluation" in proc.stdout or "Evaluation" in proc.stdout:
            print("   ✓ Evaluation works")
            # Extract eval
            for line in proc.stdout.split('\n'):
                if 'evaluation' in line.lower() or 'score' in line.lower():
                    print(f"   {line.strip()}")
        else:
            print(f"   ✗ Evaluation failed: {proc.stdout}")
            return False
    except Exception as e:
        print(f"   ✗ Evaluation test error: {e}")
        return False

    # Test 4: Search
    print("\n4. Testing search...")
    try:
        proc = subprocess.run(
            [str(stockfish_path)],
            input=f"uci\nsetoption name EvalFile value {net_path}\nisready\nposition fen {fen}\ngo depth 15\nquit\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        if "bestmove" in proc.stdout:
            print("   ✓ Search works")
            for line in proc.stdout.split('\n'):
                if 'bestmove' in line:
                    print(f"   {line.strip()}")
        else:
            print(f"   ✗ Search failed: {proc.stdout}")
            return False
    except Exception as e:
        print(f"   ✗ Search test error: {e}")
        return False

    return True


def test_cutechess_match(net_path: Path, baseline_net: Path = None, stockfish_path: Path = Path("stockfish")):
    """Run a quick cutechess-cli match vs baseline (if available)."""
    try:
        import cutechess
    except ImportError:
        print("\n5. Skipping cutechess match (not installed)")
        print("   Install with: pip install cutechess-cli")
        return

    print("\n5. Running cutechess match...")
    # This would require cutechess-cli installed and a baseline net
    # Placeholder for full implementation
    print("   Not fully implemented yet")


def main():
    parser = argparse.ArgumentParser(description="Test UCI integration with exported .nnue")
    parser.add_argument("--net", required=True, type=Path, help="Path to exported .nnue file")
    parser.add_argument("--stockfish", type=Path, default=Path("stockfish"), help="Path to Stockfish binary")
    parser.add_argument("--baseline-net", type=Path, help="Baseline net for cutechess match")
    parser.add_argument("--quick", action="store_true", help="Quick test only (no search)")

    args = parser.parse_args()

    print("=" * 60)
    print("NNUE UCI Integration Test")
    print("=" * 60)

    success = test_stockfish_eval(args.net, args.stockfish)

    if not args.quick and args.baseline_net:
        test_cutechess_match(args.net, args.baseline_net, args.stockfish)

    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()