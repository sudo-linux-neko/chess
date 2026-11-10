# NNUE Trainer Design Document

**Copyright (C) 2026 Vaibhav**  
**License: GPL-3.0-or-later**

## 1. Architecture Choice

### 1.1 Feature Set: HalfKAv2_hm

We adopt the **HalfKAv2_hm** feature set from Stockfish's NNUE architecture. This is a proven, strong feature set that balances expressiveness and training speed.

- **HalfKAv2**: King-square-indexed feature set with 64 king buckets. Each bucket contains:
  - Own pieces (10 piece types × 64 squares = 640)
  - Enemy pieces (10 piece types × 64 squares = 640)
  - Total per bucket: 1,280 input features
- **hm (history/mixed)**: Adds 7 previous moves' worth of features (simplified to recent move history)

**Input dimension**: 64 (king squares) × 1,280 = **81,920** (sparse, ~30-40 active per position)

### 1.2 Network Architecture

We use a **small, fast-to-train network** optimized for ~10 hours on 2× T4:

| Layer | Size | Activation | Notes |
|-------|------|------------|-------|
| Input | 81,920 (sparse) | — | HalfKAv2_hm |
| Feature Transformer (FT) | 2 × 256 | ClippedReLU | Per-king-bucket, then concatenated |
| L1 | 32 | ClippedReLU | |
| L2 | 32 | ClippedReLU | |
| Output | 1 | Linear | Scalar evaluation (centipawns) |

**Total parameters**: ~2.1M (trainable)  
**Why this size**:
- Fits comfortably in T4 16 GB VRAM with batch size 16,384 (2×8,192 per GPU)
- Forward/backward pass ~2-3ms per batch on T4 → ~500-800k positions/sec/GPU
- 10 hours × 2 GPUs × 600k pos/s ≈ **43 billion positions** — sufficient for convergence
- Matches "small net" configs used in Stockfish nnue-pytorch training runs

**Quantization target**: 8-bit weights (Stockfish standard), 16-bit accumulator

### 1.3 Loss Function

**WDL + Eval blended loss** (Stockfish standard):

```
L = λ * L_WDL + (1 - λ) * L_Eval
```

- `L_WDL`: Binary cross-entropy on {Win, Draw, Loss} targets (3-class)
- `L_Eval`: MSE on centipawn evaluation targets
- `λ = 0.5` (configurable, tuned for strength/hour)

**Score scaling**: Sigmoid mapping from centipawns to WDL probabilities using `Q = 370` (Stockfish default)

## 2. Data Choice

### 2.1 Source: Leela Chess Zero / Stockfish Training Data

Primary source: **Hugging Face `leela-chess/lichess-binned`** or Stockfish's official training data mirrors.

- Format: `.binpack` (nnue-pytorch compatible)
- Each file: ~1-2 GB, ~10-20M positions
- Legally redistributable: Lichess games (CC0), Stockfish self-play (GPL-compatible)

### 2.2 Data Pipeline

- **Streaming loader**: C++ binpack loader from nnue-pytorch (compiled at setup)
- **Multi-worker**: 4-8 CPU workers per GPU, prefetch factor 4
- **Shuffling**: In-file shuffle + inter-file rotation (buffer 1M positions)
- **RAM usage**: <2 GB (streaming, no full load)
- **Disk cap**: 15 GB max in `/kaggle/working/data/` (configurable)

### 2.3 Download Strategy

| Priority | Source | URL Pattern |
|----------|--------|-------------|
| 1 | Hugging Face | `hf://leela-chess/lichess-binned/*.binpack` |
| 2 | Stockfish mirrors | `https://trainingdata.stockfishchess.org/*.binpack` |
| 3 | Fallback | User-provided local path |

- Resume via `hf_hub_download` (HF) or HTTP Range requests
- SHA256 checksum verification per file
- Auto-cleanup old files if disk > 18 GB

## 3. Time Budget Math (10 Hours on 2× T4)

| Component | Time Estimate | Notes |
|-----------|---------------|-------|
| Setup (deps, compile, download) | 30 min | One-time, cached across sessions |
| Data download (15 GB) | 15 min | ~150 MB/s on Kaggle |
| Training (9.5 h) | 570 min | Target |
| Checkpoint/upload overhead | 15 min | 20-min intervals, background |
| **Total** | **~10 h** | Fits session limit |

### 3.1 Throughput Targets

- **Batch size**: 16,384 (8,192 per GPU)
- **Positions/sec/GPU**: ~600k (mixed precision fp16)
- **Effective throughput**: ~1.2M pos/sec (2 GPUs)
- **Positions in 9.5 h**: 9.5 × 3600 × 1.2M ≈ **41 billion**
- **Epochs**: ~2-3 over 15B unique positions (with reshuffling)

### 3.2 Learning Rate Schedule (Time-Aware)

```
total_steps = (budget_seconds - setup_seconds) * positions_per_second / batch_size
lr_schedule = CosineAnnealingLR(eta_min=lr * 0.01, T_max=total_steps)
```

- Warmup: 1% of steps (linear)
- Decay: Cosine to 1% of peak LR
- Peak LR: 8e-4 (Ranger/AdamW, tuned for HalfKAv2)

## 4. Training Configuration (YAML)

All hyperparameters in `configs/train.yaml`:

```yaml
# Hardware
gpus: 2
precision: fp16
num_workers: 8
prefetch_factor: 4

# Model
feature_set: HalfKAv2_hm
ft_size: 256
hidden_sizes: [32, 32]
output_size: 1

# Training
batch_size: 16384
max_positions: 41000000000
learning_rate: 8e-4
optimizer: ranger  # or adamw
weight_decay: 1e-4
wdl_lambda: 0.5
score_scale: 370.0

# Time budget
time_budget_hours: 10.0
checkpoint_interval_min: 20

# Data
data_dir: /kaggle/working/data
max_data_gb: 15
data_sources:
  - hf: leela-chess/lichess-binned
  - http: https://trainingdata.stockfishchess.org

# Checkpointing
checkpoint_dir: /kaggle/working/checkpoints
keep_last_n: 3
hf_repo: vaibhav/nnue-trainer-checkpoints
hf_token_env: HF_TOKEN

# Export
export_dir: /kaggle/working/export
quantize: true
verify_tolerance: 0.01  # centipawns
```

## 5. Checkpointing & Resume Strategy

### 5.1 Checkpoint Contents

```python
{
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'scaler': scaler.state_dict(),
    'step': global_step,
    'epoch': current_epoch,
    'rng_states': {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    },
    'data_position': data_loader.get_position(),  # file + offset
    'config': config_dict,
    'wall_time_elapsed': seconds,
}
```

### 5.2 Atomic Save

```python
tmp_path = checkpoint_dir / f"step_{step}.tmp"
torch.save(state, tmp_path)
tmp_path.rename(checkpoint_dir / f"step_{step}.ckpt")
```

### 5.3 HF Upload (Background Thread)

- Queue-based uploader (non-blocking)
- Retry: exponential backoff (1s, 2s, 4s, 8s, max 60s)
- Max 3 concurrent uploads
- Never crashes training on failure (logs warning only)

### 5.4 Resume Logic

1. Check `/kaggle/working/checkpoints/` for latest `step_*.ckpt`
2. If none, query HF repo `vaibhav/nnue-trainer-checkpoints` for latest
3. Download to local, verify checksum
4. Load full state including data position
5. Continue exactly from interrupted step

### 5.5 SIGTERM Handling

- Register signal handler for SIGTERM/SIGINT
- On signal: set `should_stop = True`, finish current batch, save final checkpoint, upload, exit(0)

## 6. Export & UCI Usability

### 6.1 Export Process

1. Load best checkpoint (lowest validation loss)
2. Convert to eval mode, fuse BatchNorm (if any)
3. Quantize weights to int8 (per-channel symmetric)
4. Serialize to Stockfish `.nnue` format (compatible with `serialize.py`)
5. Save to `/kaggle/working/export/nn-<hash>.nnue`

### 6.2 Verification

- Sample 10,000 test positions from held-out data
- Compare PyTorch model output vs. exported net (via Stockfish `EvalFile`)
- Tolerance: ≤0.01 centipawns mean absolute error
- Fail export if tolerance exceeded

### 6.3 Stockfish Integration

- **Compatible commit**: `stockfish/stockfish@nnue-pytorch` branch or main branch after NNUE merge
- **Build script**: `scripts/build_stockfish.sh` (downloads, patches, builds)
- **Test script**: `scripts/test_uci.py` (runs `EvalFile`, cutechess match vs baseline)

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Kaggle GPU quota exhausted | Medium | High | Detect early, fallback to CPU smoke test |
| Disk full (/kaggle/working 20 GB) | Medium | High | Auto-cleanup, configurable caps, monitor |
| Data download fails | Low | Medium | Multiple sources, resume, local fallback |
| HF upload fails (token/network) | Medium | Low | Background, retry, never block training |
| OOM on T4 16 GB | Low | High | Batch size auto-tune, gradient accumulation |
| Training diverges | Low | High | Gradient clipping (1.0), LR warmup, loss monitoring |
| Session killed before checkpoint | Medium | Medium | 20-min intervals + SIGTERM handler |
| Quantization accuracy loss | Low | Medium | Per-channel quantization, verification step |
| Stockfish version mismatch | Low | Medium | Pin exact commit, document compatibility |

## 8. File Structure

```
nnue-trainer/
├── LICENSE
├── NOTICE
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── train.yaml
├── src/
│   ├── __init__.py
│   ├── model.py          # HalfKAv2_hm network
│   ├── data.py           # Binpack dataset + loader
│   ├── train.py          # Training loop, Lightning module
│   ├── checkpoint.py     # Save/load/resume logic
│   ├── hf_upload.py      # Background HF uploader
│   ├── export.py         # Quantization + .nnue serialization
│   ├── verify.py         # Export verification
│   └── cli.py            # run.py entry point
├── scripts/
│   ├── setup_and_train.sh
│   ├── build_stockfish.sh
│   ├── test_uci.py
│   └── publish_github.sh
├── notebooks/
│   └── kaggle_train.ipynb
├── tests/
│   ├── test_checkpoint.py
│   ├── test_atomic_write.py
│   ├── test_export.py
│   └── test_smoke.py
└── docs/
    └── DESIGN.md
```

## 9. One-Command UX

### 9.1 `python run.py` (or `bash setup_and_train.sh`)

```bash
# Does all of:
1. pip install -r requirements.txt
2. cmake + make for C++ binpack loader (src/cpp_loader/)
3. Download data (resume, checksum, cap)
4. HF auth from HF_TOKEN secret/env
5. Detect latest checkpoint (local → HF)
6. Train on 2 GPUs (DDP/torchrun)
7. Checkpoint every 20 min (atomic + HF upload)
8. Export final .nnue + verify
9. Print summary + UCI test instructions
```

### 9.2 `--smoke-test` Flag

```bash
python run.py --smoke-test
# Runs 2 min on 1000 positions, fake HF upload, tests resume
```

### 9.3 Kaggle Notebook (`notebooks/kaggle_train.ipynb`)

3 cells:
1. `!git clone <repo> && cd nnue-trainer`
2. Set `HF_TOKEN` in Kaggle Secrets
3. `!python run.py` (or `!bash setup_and_train.sh`)

---

*End of DESIGN.md*