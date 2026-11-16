# Running on free Kaggle 2x T4 (9 hours)

## 1. Notebook settings

- Accelerator: **GPU T4 x2**
- Internet: **On**
- Persistence: *Files only*

Free GPU quota is about 30 h/week; one full run is 9 h.

## 2. Install

```bash
!git clone https://github.com/thisisforlearn/chess-works.git
%cd chess-works

!pip install -q maturin huggingface_hub pyyaml
!maturin develop --release -m crates/nnue_core/Cargo.toml
```

Kaggle already ships CUDA PyTorch and numpy. `maturin develop` builds the Rust
extension (`nnue_core`) against the active Python and can take a few minutes.

## 3. Optional: Hugging Face upload

Add a Kaggle secret `HF_TOKEN` (Add-ons -> Secrets), then:

```python
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
```

Set in `configs/train.yaml`:

```yaml
checkpoint:
  hf_repo: thisisforlearn/chess-works-nnue
  upload: true
  resume: true
```

With `resume: true`, a missing local checkpoint is downloaded from the repo,
so a restarted session continues from the last uploaded step. Uploads run in a
background thread and never block training.

## 4. Train

```bash
!python run.py train --config configs/train.yaml --kaggle
```

`--kaggle` starts `torch.distributed.run` with 2 worker processes (one per T4).
The trainer:

1. downloads `fishpack32.relabel-BT4-tf13tune.binpack` (5.9 GB) to
   `/kaggle/working/data`;
2. reads two disjoint, chunk-aligned shards at ~10 Mpos/s per rank;
3. measures ~30 steps, then auto-sizes `total_steps` to fit
   `train.time_budget_hours`;
4. writes `checkpoints/latest.pt` every 10 minutes (atomic);
5. exports `checkpoints/final.nnue` and verifies it against the float model.

Adjust `train.batch_size` if you hit OOM. 16384 per GPU is conservative for
`L1=128`.

## 5. Finish and download

At the end of the run:

```
checkpoints/latest.pt   resumable state
checkpoints/final.pt    final checkpoint
checkpoints/final.nnue  Stockfish 16.1 Small net (~3.1 MB)
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
- The budget includes data download and the ~1 min export/verify. Leave
  `time_budget_hours` at 9.0–9.3, not 12.
- The output architecture is **Stockfish 16.1 Small**, not modern master.
