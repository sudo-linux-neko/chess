# NNUE Trainer for Stockfish

**Copyright (C) 2026 Vaibhav**  
**License: GPL-3.0-or-later** — see [LICENSE](LICENSE) and [NOTICE](NOTICE)

A production-quality Stockfish-style NNUE trainer that runs on **Kaggle's FREE tier** (2× NVIDIA T4, 16 GB each, ~12h session) and produces a `.nnue` network usable in a UCI engine.

---

## 🚀 Quickstart on Kaggle

### Prerequisites
- Kaggle account with GPU quota (2× T4)
- Hugging Face account + token (for checkpoint uploads)

### 1. Clone & Run (3 cells in Kaggle Notebook)

**Cell 1 — Clone:**
```bash
!git clone https://github.com/vaibhav/nnue-trainer.git
%cd nnue-trainer
```

**Cell 2 — Set Secrets:**
- Go to Kaggle sidebar → **Secrets** → **Add new secret**
- Name: `HF_TOKEN` | Value: `hf_xxxxxxxxxxxxx` (your HF write token)

**Cell 3 — Train:**
```bash
!python run.py
```
That's it. The script will:
1. Install dependencies (`requirements.txt`)
2. Compile the fast C++ binpack loader
3. Download training data (resume + checksum)
4. Authenticate to Hugging Face
5. Resume from latest checkpoint (local → HF)
6. Train on both T4s for ~10 hours
7. Checkpoint every 20 min (atomic + background HF upload)
8. Export final `.nnue` + verify accuracy

### 2. Using the Notebook (Alternative)
Open `notebooks/kaggle_train.ipynb` — it has the same 3 cells pre-filled.

### 3. Smoke Test (2 minutes)
```bash
python run.py --smoke-test
```
Verifies the entire pipeline end-to-end on a tiny data slice.

---

## 📊 Expected Results

| Met | Target |
|-----|--------|
| Training time | ~9.5 hours (of 10h budget) |
| Positions trained | ~40 billion |
| Throughput | ~1.2M positions/sec (2× T4) |
| Final net size | ~2.1M params → ~5 MB `.nnue` |
| Strength gain | +50-100 Elo vs. baseline (est.) |

---

## 🔄 Resume Across Sessions

**Just re-run the same command.** The trainer:
1. Checks `/kaggle/working/checkpoints/` for latest `step_*.ckpt`
2. If missing, downloads from `vaibhav/nnue-trainer-checkpoints` on HF
3. Restores **exact state**: model, optimizer, scheduler, RNG, data position
4. Continues training seamlessly

---

## 📁 Project Structure

```
nnue-trainer/
├── LICENSE, NOTICE, README.md
├── requirements.txt, .gitignore
├── configs/
│   └── train.yaml          # All hyperparameters
├── src/
│   ├── model.py            # HalfKAv2_hm network
│   ├── data.py             # Binpack dataset + loader
│   ├── train.py            # Training loop (Lightning)
│   ├── checkpoint.py       # Save/load/resume
│   ├── hf_upload.py        # Background HF uploader
│   ├── export.py           # Quantize + .nnue export
│   ├── verify.py           # Export verification
│   └── cli.py              # run.py entry point
├── scripts/
│   ├── setup_and_train.sh  # One-command wrapper
│   ├── build_stockfish.sh  # Build matching Stockfish
│   ├── test_uci.py         # UCI/EvalFile test
│   └── publish_github.sh   # Publish to GitHub (gh CLI)
├── notebooks/
│   └── kaggle_train.ipynb  # Ready-to-run Kaggle notebook
├── tests/
│   ├── test_checkpoint.py
│   ├── test_atomic_write.py
│   ├── test_export.py
│   └── test_smoke.py
└── docs/
    └── DESIGN.md           # Architecture & design decisions
```

---

## ⚙️ Configuration

All settings in `configs/train.yaml`:

```yaml
gpus: 2
precision: fp16
batch_size: 16384
learning_rate: 8e-4
optimizer: ranger
wdl_lambda: 0.5
time_budget_hours: 10.0
checkpoint_interval_min: 20
max_data_gb: 15
hf_repo: vaibhav/nnue-trainer-checkpoints
```

Override via CLI:
```bash
python run.py --config configs/train.yaml --batch_size 8192 --time_budget_hours 5
```

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/ -v

# Smoke test (2 min)
python run.py --smoke-test
```

---

## 📦 Export & UCI Usage

After training, the final net is at:
```
/kaggle/working/export/nn-<hash>.nnue
```

### Use with Stockfish:

```bash
# Build matching Stockfish (one-time)
./scripts/build_stockfish.sh

# Quick eval test
./scripts/test_uci.py --net /kaggle/working/export/nn-*.nnue

# Play a game (cutechess)
cutechess-cli -engine cmd=stockfish name=NewNet option.EvalFile=/path/to/nn-*.nnue \
              -engine cmd=stockfish name=Baseline \
              -rounds 10 -tc 10/1
```

**Compatible Stockfish commit**: `stockfish/stockfish@master` (post-NNUE merge, tested with `nnue-pytorch` branch)

---

## 🛠 Troubleshooting

| Issue | Fix |
|-------|-----|
| "No GPU available" | Kaggle → Settings → GPU → T4 × 2 |
| "Disk full" | Reduce `max_data_gb` in config; auto-cleanup runs |
| "HF upload failed" | Check `HF_TOKEN` secret; uploads are non-blocking |
| "OOM on T4" | Reduce `batch_size` (auto-tune: `--batch_size 8192`) |
| "Session killed" | Re-run command; resumes from last checkpoint |
| "Import error (cpp_loader)" | Run `python run.py` again — compiles on first run |

---

## 📄 License

GNU GPL v3.0 — see [LICENSE](LICENSE).  
This project builds on [official-stockfish/nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) (GPL-3.0).

---

## 🙏 Acknowledgments

- Stockfish developers for NNUE architecture & training code
- Leela Chess Zero for training data
- Kaggle for free GPU compute
- Hugging Face for model hosting