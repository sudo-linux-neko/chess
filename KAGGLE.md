# Running on free Kaggle 2x T4 (9 hours)

## 1. Notebook settings

- Accelerator: **GPU T4 x2**
- Internet: **On**
- Persistence: *Files only*

Free GPU quota is about 30 h/week; one full run is 9 h.

## 2. Install

The easiest path is the single self-contained cell in `kaggle_single_cell.py`
(it installs Rust if missing, clones/pulls, builds a wheel, logs in, trains,
and prints the result). Manual equivalent:

```bash
!git clone https://github.com/thisisforlearn/chess-works.git
%cd chess-works

!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
!source $HOME/.cargo/env && pip install -q "maturin>=1.7" "huggingface_hub>=0.23" pyyaml
!source $HOME/.cargo/env && maturin build --release -m crates/nnue_core/Cargo.toml
!pip install -q --force-reinstall --no-deps crates/nnue_core/target/wheels/nnue_core-*.whl
```

Kaggle already ships CUDA PyTorch and numpy. Building the wheel avoids
`maturin develop` requiring a virtualenv.

## 3. Hugging Face upload (default on)

Add a Kaggle secret named `HF_TOKEN` (Add-ons -> Secrets) with a **write** token.
The trainer reads it automatically, and this cell does the same explicitly:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami

user_secrets = UserSecretsClient()
login(token=user_secrets.get_secret("HF_TOKEN"), add_to_git_credential=False)
print("HF account:", whoami()["name"])
```

The account name is read from the token, never hardcoded. The trainer prints
`[hf] logged in as <account>` and checks that the target repo owner matches
that account (or one of its orgs).

Defaults in `configs/train.yaml`:

```yaml
checkpoint:
  hf_repo: neko-i-am-a-cat/chess-nnue
  upload: true
  resume: true
```

Uploads (background thread, never block training):

```
checkpoints/step_0001234_elapsed_02h15m_left_06h45m_loss_0.02413_up_007.pt
latest.pt
training_state.json
final.pt
final.nnue
```

With `resume: true`, a missing local checkpoint is downloaded from the repo,
so a restarted session continues from the last uploaded step.

## 4. Train

```bash
!python run.py train --config configs/train.yaml --kaggle
```

`--kaggle` starts `torch.distributed.run` with 2 worker processes (one per T4).
The trainer:

1. downloads two binpacks (~13.7 GB total: `fishpack32` and
   `wrongIsRight_nodes5000pv2`) to `/kaggle/working/data`;
2. reads two disjoint, chunk-aligned shards at ~10 Mpos/s per rank, with each
   rank starting on a different pack so both datasets mix from step one;
3. warms up 50 steps, measures 200 steps, then auto-sizes `total_steps` and
   periodically extends it so the full `train.time_budget_hours` is used;
4. writes a descriptive checkpoint every 10 minutes (atomic), keeping only
   ~2 GB locally;
5. exports `checkpoints/final.nnue` and verifies it against the float model.

`train.compile: true` uses `torch.compile` (eager fallback on failure).
`train.cpu_threads: 0` picks 1 Torch thread per DDP rank on 4 vCPUs, leaving
cores for the Rust I/O/decode threads.

For a clean restart, set `FRESH = True` in the self-contained cell: it deletes
`/kaggle/working/chess-works/checkpoints` and sets `NNUE_NO_RESUME=1`, so the
trainer ignores `latest.pt` on Hugging Face too. Set `FRESH = False` only when
you want to resume from the last uploaded checkpoint.

Adjust `train.batch_size` if you hit OOM. 16384 per GPU is conservative for
`L1=128`.

## 5. Finish and download

At the end of the run:

```
checkpoints/step_*_elapsed_*_left_*_loss_*_up_*.pt   periodic checkpoints
checkpoints/latest.pt                              resumable alias
checkpoints/final.pt                               final checkpoint
checkpoints/final.nnue                             Stockfish 16.1 Small net
checkpoints/training_state.json                    account/step/loss/ETA
```

Copy `final.nnue` to `/kaggle/working/` (or let the HF uploader handle it) and
download it from the notebook output.

## 6. Verify with Stockfish 16.1

```bash
!git clone --depth 1 --branch sf_16.1 https://github.com/official-stockfish/Stockfish.git /tmp/sf
!cd /tmp/sf/src && make -j2 build ARCH=x86-64-avx2
!printf 'setoption name EvalFileSmall value /kaggle/working/chess-works/checkpoints/final.nnue\nposition startpos\neval\nquit\n' | /tmp/sf/src/stockfish
```

A normal Stockfish 16.1 binary only selects the small net in lopsided
positions. `scripts/build_stockfish.sh` builds a parity binary that forces the
small net in `eval`.

## Notes

- Only ~7.6 GB system RAM locals are irrelevant here; Kaggle T4 workers have
  more. The Rust stream keeps memory bounded (a few MB of chunks).
- The budget includes data download, compile startup, and export/verify. Leave
  `time_budget_hours` at 9.0–9.3, not 12.
- The two binpacks use ~12.7 GiB; checkpoints add ~2 GiB. Keep the notebook
  output disk above ~20 GiB free (the UI may show a larger max).
- The output architecture is **Stockfish 16.1 Small**, not modern master.
