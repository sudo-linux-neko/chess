# chess-works

A fast, correct trainer for **Stockfish 16.1 "Small"** NNUE networks
(`HalfKAv2_hm`, `L1=128`, `L2=15`, `L3=32`, 8 layer stacks), built for the free
Kaggle 2x T4 tier (9 hour sessions).

Author: **Vaibhav** (`thisisforlearn`)
License: **GPL-3.0-or-later** (see `LICENSE`)

## Why this exists

The original repository was a mock: it invented a fake `BINP` format, fake
`NNUE` export, and the wrong model dimensions, so it could never produce a net
Stockfish would load. This rewrite does the real thing:

- **Rust core** (`crates/nnue_core`, PyO3):
  - real Stockfish `.binpack` decoding (`sfbinpack`);
  - exact `HalfKAv2_hm` feature extraction, 22528 inputs, 32 slots per
    perspective, `PAD_INDEX=22528`;
  - threaded I/O + decode pipeline (**~10 M positions/s per process**);
  - exact quantized `.nnue` writer and exact integer inference.
- **PyTorch** (`src/`): only the GPU training loop, loss, checkpointing and
  export orchestration.
- **Verified against Stockfish 16.1**: the generated net loads with
  `setoption name EvalFileSmall value final.nnue`, and `nnue_core.eval_nnue`
  matches Stockfish's small-net evaluation within UCI rounding on tested FENs.

The exported file is a **Stockfish 16.1 Small** net. Modern Stockfish master
uses a different architecture (FullThreats / `L1=1024` / 192 stacks); that is
out of scope for a 9 hour T4 run.

## Install

Linux, Rust toolchain, Python >= 3.9.

```bash
python -m venv ~/ml_env && source ~/ml_env/bin/activate
pip install -r requirements.txt
maturin develop --release -m crates/nnue_core/Cargo.toml
```

`maturin develop` installs the `nnue_core` extension into the active venv.
After changing Rust code, re-run it.

## Data

Real binpacks from the `vondele` Hugging Face datasets. The default config uses
two packs (~13.7 GB total) for better diversity while staying inside a 20 GB
notebook disk:

```yaml
data:
  repo: vondele/master-binpacks_relabel
  files:
    - fishpack32.relabel-BT4-tf13tune.binpack
    - wrongIsRight_nodes5000pv2.relabel-BT4-tf13tune.binpack
```

Files are downloaded once into `cache_dir` (`/kaggle/working/data` on Kaggle,
`./data` locally). Ranks read disjoint, chunk-aligned shards.

## Train

Single process / CPU smoke test:

```bash
python run.py train --config configs/smoke.yaml
```

2x T4 on Kaggle:

```bash
python run.py train --config configs/train.yaml --kaggle
```

`--kaggle` launches `torch.distributed.run` with one process per GPU. You can
also launch DDP yourself:

```bash
torchrun --nproc_per_node=2 run.py train --config configs/train.yaml
```

The run is **time-budgeted** and stops at the wall-clock limit as a safety
net. Set `train.max_steps` to force a fixed number of steps.

Auto-sizing warms up for 50 steps, measures 200 steps with a GPU sync, and
periodically extends `total_steps` if throughput rises, so the wall-clock
budget is actually used. On a fresh multi-GPU run each rank starts on a
different binpack, mixing datasets immediately. Checkpoints are atomic and
store model, optimizer, step, LR schedule, RNG and each rank's exact binpack
cursor `(file, byte, skip_entries)`. Resume is
bit-exact. Every periodic checkpoint is named with its step, elapsed run time,
budget left, loss and upload number, e.g.
`step_0001234_elapsed_02h15m_left_06h45m_loss_0.02413_up_007.pt`.

Upload to the Hugging Face Hub is **on by default**:

```yaml
checkpoint:
  hf_repo: neko-i-am-a-cat/chess-nnue
  upload: true
  resume: true
```

The token is resolved from `HF_TOKEN`, then the Kaggle secret named `HF_TOKEN`,
then the local `hf auth login`. The real account name is printed (from
`whoami()`), and the repo owner is checked against it. Each checkpoint uploads
the named file, `latest.pt`, and `training_state.json`; the run also uploads
`final.pt` and `final.nnue`. Uploads run in a background thread.

## Export and verify

```bash
python run.py export --checkpoint checkpoints/latest.pt --out final.nnue
python run.py info --nnue final.nnue
python run.py verify --checkpoint checkpoints/latest.pt --nnue final.nnue \
    --binpack data/fishpack32.relabel-BT4-tf13tune.binpack
```

`verify` compares the float model (`pred * 600`) against Stockfish's exact
integer inference. Quantization error on a random network is ~MAE 7 internal
units (~2 cp) with correlation > 0.999.

Load it in a real Stockfish 16.1:

```bash
git clone --depth 1 --branch sf_16.1 https://github.com/official-stockfish/Stockfish.git
cd Stockfish/src && make -j"$(nproc)" build ARCH=x86-64-avx2
printf 'setoption name EvalFileSmall value %s\nposition startpos\neval\nquit\n' \
    "$PWD/../../final.nnue" | ./stockfish
```

The stock 16.1 binary only uses the small net in strongly unbalanced positions;
`scripts/build_stockfish.sh` also builds a debug binary whose `eval` forces the
small net, which is what `scripts/test_uci.py` uses for parity checks.

## Architecture details

- Feature set: `HalfKAv2_hm`, `HashValue=0x7f234cb8`, 22528 inputs.
- King buckets: white `(7-rank)*4 + [0,1,2,3,3,2,1,0][file]`, black analogous;
  32 buckets per perspective. Layer-stack bucket `(piece_count-1)//4`, 0..7.
- FT: shared table, `128 + 8` outputs (128 positional + 8 PSQT buckets),
  int16 quantization `*127`.
- Pairwise product: `(us[i]*us[i+64])` and `(them[i]*them[i+64])`, scaled
  `127/128`, then `32 -> 16 -> [sqr 15 | clipped 15] -> 32 -> 1`, plus skip.
- PSQT: `(w_psqt - b_psqt) * (us - 0.5)`, quantized `*9600`.
- Hash: `version=0x7AF32F20`, `ft_hash = 0x7f234cb8 ^ (L1*2)`,
  `net_hash = fc_hash ^ ft_hash`.

## Performance notes

Measured on an i5-1335U / WSL2 (7.6 GB RAM), real `tb6dtm.binpack`:

- Rust reader + feature extraction: **9.9–10.2 M positions/s** sustained.
- The initial `rayon` version was *slower* (3.5 Mpos/s with 12 threads) due to
  contention; a single serial decoder behind a bounded I/O thread wins.
- Python only receives `int16` feature indices; no Python-side decoding.

## Tests

```bash
python -m pytest -q tests
(cd crates/nnue_core && cargo test --release)
```

## Layout

```
crates/nnue_core/   Rust: binpack stream, features, .nnue writer + eval
src/                Python: model, loss, data bridge, trainer, export, verify
configs/            train.yaml (Kaggle), smoke.yaml (CPU)
scripts/            setup, Stockfish build/parity, publish helper
tests/              pytest suite for features, model, export, resume
```
