# Design

## Split

```
.binpack --[Rust: I/O thread]--> raw chunks --[Rust: decoder]--> Sample blocks
   --> numpy int16 features --> [PyTorch: GPU] --> float net
   --> [Rust: quantizer/writer] --> Stockfish 16.1 Small .nnue
```

Python never parses binpacks or computes feature indices. One process keeps
two Rust threads and two bounded channels (I/O depth 3, decoded depth 6), so
disk reads overlap chunk parsing without unbounded memory.

## Sample

```rust
struct Sample {
    features: [i16; 64],  // white 32 | black 32, padded with 22528
    n: [u8; 2],           // active features per perspective
    stm: u8,              // 0 white, 1 black
    bucket: u8,           // (piece_count - 1) / 4
    score: i16,           // side-to-move relative, Stockfish units
    result: i8,           // 1 / 0 / -1, side-to-move relative
    ply: u16,
    chunk_start: u64,
    entry_idx: u32,
}
```

`(chunk_start, entry_idx)` is an exact resume cursor. `find_chunk_offset`
snaps a byte estimate to a `BINP` chunk header, so DDP ranks read disjoint
chunk-aligned shards.

## Float model (matches Stockfish integer inference)

- `l0 = us*[w,b] + them*[b,w]`, clamp `[0,1]`.
- `pair = [us[:64]*us[64:], them[:64]*them[64:]] * 127/128`.
- Per-sample layer stack `bucket`: `128 -> 16`, `c = out[:15]`,
  `skip = out[15]`, `[c*c*127/128, c] -> 32`, clamp, `-> 1`.
- `psqt = (w_psqt - b_psqt) * (us - 0.5)`, 8 PSQT buckets.
- Output in units where `1.0 == 600` (Stockfish internal).

Weights are clipped to exactly representable ranges after every optimizer
step: hidden `|w| <= 127/64`, output `|w| <= 127*127/9600`, FT int16.

## Quantization / export

```
FT weight,bias   *127         int16
PSQT             *9600        int32
hidden weight    *64          int8   clamp +/-127/64
hidden bias      *8128        int32
output weight    *9600/127    int8   clamp +/-127*127/9600
output bias      *9600        int32
fc0[15] (skip)   *9600/8128   folded into output bias
```

`nnue_core.write_nnue` emits signed LEB128 blocks, the 16.1 header and hashes,
and 8 layer stacks. `nnue_core.eval_nnue` is an independent exact integer
implementation used by tests and `run.py verify`.

## Training

- Loss (nnue-pytorch `step_`): win-rate blend, `|pt-qf|^2.5`.
- Time-budgeted: measure steps/s after 30 steps, set `total_steps` from the
  remaining wall clock, broadcast to all ranks.
- Checkpoints: atomic replace, model+optimizer+step+cursors+RNG; background
  HF upload.
- DDP: `torch.distributed.run`, one rank per T4; all-reduce via NCCL.

## Verification

1. `pytest`: features, model gradients/clipping, quantization shapes,
   export-vs-float parity, exact resume, checkpoint roundtrip.
2. `cargo test`: dims, buckets, LEB128, SF16.1 hashes.
3. Real Stockfish 16.1 loads `EvalFileSmall` and `eval` matches `eval_nnue`
   within UCI rounding on a set of FENs.
