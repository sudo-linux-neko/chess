# Kaggle Training Guide: Ultra-Optimized NNUE Trainer

**Target**: Kaggle Free Tier — 2× T4 GPUs (16 GB each), 12h session, ~20 GB `/kaggle/working`

**Expected Result**: ~3750-3780 CCRL Elo net in ~10 hours

---

## 📋 Prerequisites

1. **Kaggle Account** — [kaggle.com](https://kaggle.com)
2. **Hugging Face Token** (for checkpoint uploads) — [hf.co/settings/tokens](https://huggingface.co/settings/tokens) with `write` permission
3. **GPU Quota** — Verify you have T4 x2 available (Settings → GPU)

---

## 🚀 Copy-Paste Commands

### Step 1: Clone & Install (Run in Cell 1)

```python
# ============================================================
# ULTRA-OPTIMIZED NNUE TRAINER - KAGGLE 2x T4 (10h)
# ============================================================
!git clone https://github.com/sudo-linux-neko/chess.git
%cd chess

# Install deps (includes PyTorch 2.3+ with CUDA 12.1)
!pip install -q -r requirements.txt 2>&1 | tail -5

# Verify GPU
!nvidia-smi -L
import torch; print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPUs: {torch.cuda.device_count()}")
```

**Wait**: ~2-3 minutes

---

### Step 2: Add HF Token (Required)

1. **Settings** (gear icon ☰) → **Secrets** → **Add new secret**
2. **Label**: `HF_TOKEN`
3. **Value**: Your Hugging Face token (needs `write` permission)
4. **Save**

---

### Step 3: Download Data + Compile Loader (Run in Cell 2)

```python
# ============================================================
# PREP: Download high-quality Stockfish data + compile loader
# ============================================================
!python run.py --config configs/train_ultra.yaml --download-data 2>&1 | tail -20
!python run.py --config configs/train_ultra.yaml --compile-only 2>&1 | tail -10
```

**Wait**: ~5-15 minutes
- Downloads ~18 GB from `vondele/*` HF datasets (Stockfish self-play, Leela games, tablebase endgames)
- Compiles C++ binpack loader (pybind11)

---

### Step 4: MAIN TRAINING — 9.5 Hours (Run in Cell 3)

```python
# ============================================================
# TRAIN: Full ultra-optimized run (9.5 hours on 2x T4)
# ============================================================
!python run.py --config configs/train_ultra.yaml 2>&1 | tee training.log
```

**This runs ~10 hours**. You can:
- Close browser — Kaggle keeps it running
- Check `training.log` in Files panel for progress
- Checkpoints every 10 min to `/kaggle/working/checkpoints/`
- Auto-uploads to HF Hub (if token set)

---

### Step 5: Get Your .nnue (Run After Training)

```python
# ============================================================
# EXPORT: Already done automatically, but verify manually
# ============================================================
!ls -lh /kaggle/working/export/
!python run.py --config configs/train_ultra.yaml --verify-export 2>&1 | tail -20
```

**Your net**: `/kaggle/working/export/nn-<hash>.nnue` (~10 MB)

---

### Step 6: Download & Use in Stockfish

1. In Kaggle **Files panel** (left sidebar): navigate to `working/export/`
2. Right-click `nn-*.nnue` → **Download**
3. Copy to Stockfish nnue folder:
   - **Linux**: `~/.local/share/stockfish/nnue/`
   - **Windows**: `%APPDATA%\Stockfish\nnue\`
   - **macOS**: `~/Library/Application Support/Stockfish/nnue/`
4. In Stockfish UCI: `setoption name EvalFile value nn-<hash>.nnue`

---

## 🔁 If Session Crashes / You Restart

```python
# Auto-resumes from latest checkpoint (local or HF Hub)
!python run.py --config configs/train_ultra.yaml 2>&1 | tee -a training.log
```

---

## 📋 Quick Commands Reference

| Command | Purpose |
|---------|---------|
| `--smoke-test` | 2-min pipeline verification |
| `--benchmark` | Throughput benchmark (no training) |
| `--export-only` | Export latest checkpoint to .nnue |
| `--verify-export` | Verify .nnue matches PyTorch |
| `--build-rust` | Build Rust SIMD loader (experimental) |
| `--compile-only` | Only compile C++ loader |
| `--download-data` | Only download training data |

---

## ⚠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| **C++ compile fails** | `!apt-get update && apt-get install -y cmake g++ pybind11-dev` |
| **OOM on T4** | Batch is 65K (micro 8K × 8 accum) — fits 16 GB VRAM |
| **No HF token** | Checkpoints save locally only; no cloud backup |
| **Slow download** | First run downloads ~18 GB; subsequent runs use cache |
| **Session dies at 12h** | Trainer handles SIGTERM — resumes on re-run |

---

## 📊 Expected Output

```
TRAINING COMPLETE
Final model: /kaggle/working/export/nn-a1b2c3d4.nnue
Total time: 9.7 hours
Steps: 5,108,000
Verification: PASSED (MAE=0.006 cp, Max=0.23 cp)
```

---

## 🏗️ Architecture Summary

| Component | Spec |
|-----------|------|
| **Features** | HalfKAv2_hm (64×1280 = 81,920) |
| **Network** | FT(256 shared) → L1(256) → L2(32) → Out(1) |
| **Params** | ~2.5M |
| **Batch** | 65,536 (8,192 micro × 8 accum) |
| **LR Schedule** | One-Cycle (0.0008, 5.1M steps, 5% warmup) |
| **Lambda** | Cosine jitter (δ=-0.3, sample/batch jitter) |
| **Data** | vondele T60-T70, master-binpacks, Leela, tablebase |
| **Quantization** | Per-channel symmetric int8 (Stockfish format) |

---

## 📄 License

**GPL-3.0-or-later** — Copyright (C) 2026 Vaibhav

See [LICENSE](../LICENSE) and [NOTICE](../NOTICE).