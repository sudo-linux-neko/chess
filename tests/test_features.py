# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
import nnue_core


def test_constants():
    assert nnue_core.INPUT_DIM == 22528
    assert nnue_core.MAX_ACTIVE == 32
    assert nnue_core.PAD_INDEX == 22528
    assert nnue_core.NUM_BUCKETS == 64


def test_startpos_features():
    feats, nw, nb, bucket = nnue_core.fen_features(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    assert nw == 32 and nb == 32 and bucket == 7
    assert len(feats) == 64
    assert max(feats) < nnue_core.INPUT_DIM


def test_king_bucket_e1_e8():
    _, _, _, b1 = nnue_core.fen_features("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    _, _, _, b2 = nnue_core.fen_features("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
    assert b1 == b2 == 0


def test_empty_board_has_two_kings():
    feats, nw, nb, _ = nnue_core.fen_features("8/8/8/8/8/8/8/K6k w - - 0 1")
    assert nw == 2 and nb == 2
    assert feats[2:32] == [nnue_core.PAD_INDEX] * 30
    assert feats[34:] == [nnue_core.PAD_INDEX] * 30
