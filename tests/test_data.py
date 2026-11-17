# Copyright (C) 2026 Vaibhav
# SPDX-License-Identifier: GPL-3.0-or-later
import nnue_core

from src.data import BinpackStream, StreamState, shard_start


def test_reader_batch_shape(sample_binpack):
    r = nnue_core.BinpackReader(sample_binpack)
    b = r.next_batch(1024)
    assert b["features"].shape == (1024, 64)
    assert b["features"].dtype.name == "int16"
    assert b["stm"].shape == (1024,)
    assert b["score"].shape == (1024,)
    assert b["count"] == 1024
    assert int(b["features"].max()) <= nnue_core.PAD_INDEX


def test_exact_resume(sample_binpack):
    a = nnue_core.BinpackReader(sample_binpack)
    first = a.next_batch(1000)
    cur = a.cursor()
    second = a.next_batch(1000)
    b = nnue_core.BinpackReader(sample_binpack, cur[0], cur[1])
    again = b.next_batch(1000)
    assert (second["score"] == again["score"]).all()
    assert (second["stm"] == again["stm"]).all()


def test_stream_state_and_shard(sample_binpack):
    assert shard_start(sample_binpack, 1, 0) == 0
    s = BinpackStream([sample_binpack], cycle=True)
    b = s.next_cpu(500)
    assert b["features"].shape == (500, 64)
    st = s.state_dict()
    assert set(st) == {"file_index", "byte_offset", "skip_entries"}
    s2 = BinpackStream([sample_binpack], state=StreamState.from_any(st), cycle=True)
    nxt = s.next_cpu(500)
    nxt2 = s2.next_cpu(500)
    assert (nxt["score"] == nxt2["score"]).all()
    assert (nxt["features"] == nxt2["features"]).all()
