# NNUE Trainer Design Document - Optimized

**Copyright (C) 2026 Vaibhav**  
**License: GPL-3.0-or-later**

## 1. Architecture Choice

### 1.1 Feature Set: HalfKAv2_hm

We adopt the **HalfKAv2_hm** feature set from Stockfish's NNUE architecture — the proven standard for strong chess evaluation.

- **HalfKAv2**: King-square-indexed feature set with 64 king buckets. Each bucket contains:
  - Own pieces (10 piece types × 64 squares = 640)
  - Enemy pieces (10 piece types × 64 squares = 640)
  - Total per bucket: 1,280 input features
- **hm (history/mixed)**: Adds recent move history features

**Input dimension**: 64 (king squares) × 1,280 = **81,920** (sparse, ~30-40 active per position)

### 1.2 Network Architecture — Optimized for Elo/hr

| Layer | Size | Activation | Notes |
|-------|------|------------|-------|
| Input | 81,920 (sparse) | — | HalfKAv2_hm |
| Feature Transformer (FT) | 64 × 256 (shared weights) | ClippedReLU | 16,384 accumulator |
| L1 (Hidden1) | 256 | ClippedReLU | Accumulator → 256 |
| L2 (Hidden2) | 32 | ClippedReLU | 256 → 32 |
| Output | 1 | Linear | Scalar evaluation (centipawns) |

**Total parameters**: ~2.5M (trainable)  
**Why this size**:
- L1=256 (not 32) provides sufficient capacity for complex positional evaluation
- L2=32 maintains fast inference for Stockfish
- FT=256 per bucket is Stockfish standard
- Fits comfortably in T4 16 GB VRAM with batch size 32,768 (2×16,384 per GPU)
- Forward/backward pass ~1.5-2ms per batch on T4 → ~800k-1M positions/sec/GPU
- 10 hours × 2 GPUs × 800k pos/s ≈ **57 billion positions** — sufficient for strong convergence

**Quantization target**: 8-bit weights (Stockfish standard), 16-bit accumulator

### 1.3 Initialization — Critical for Convergence

| Layer | Initialization | Scale |
|-------|----------------|-------|
| Feature Transformer | Kaiming Normal (fan_in, ReLU) | 1.0 |
| Hidden Layers | Kaiming Normal (fan_in, ReLU) | 0.5 |
| Output Layer | Kaiming Normal (fan_in, Linear) | 0.01 |

Kaiming initialization is essential for ReLU/ClippedReLU networks — Xavier causes dead neurons.

### 1.4 Loss Function — Stockfish Formulation

**WDL + Eval blended loss** with critical improvements:

```
L = λ * L_WDL + (1 - λ) * L_Eval
```

- `L_WDL`: Cross-entropy on {Win, Draw, Loss} with **label smoothing (0.05)**
- `L_Eval`: **Huber loss (δ=100 cp)** instead of MSE — robust to outliers/blunders
- `λ = 0.55` (slightly favors WDL for playing strength)
- `Q = 370` (Stockfish sigmoid scale)

**WDL Target Computation (exact Stockfish)**:
```
W = sigmoid(eval / Q)
L = sigmoid(-eval / Q)
D = 1 - W - L (clamped ≥ 0, then renormalized)
```

**WDL Logits from Eval Prediction**:
```
logit_W = eval_pred / Q
logit_D = 0
logit_L = -eval_pred / Q
```

## 2. Data Choice

### 2.1 Source: Leela Chess Zero / Stockfish Training Data

Primary source: **Hugging Face `leela-chess/lichess-binned`** (CC0 Lichess games)  
Fallback: **Stockfish training data mirrors** (GPL-compatible self-play)

- Format: `.binpack` (nnue-pytorch compatible)
- Each file: ~1-2 GB, ~10-20M positions
- Legally redistributable

### 2.2 Data Pipeline — Optimized for Throughput

- **Streaming loader**: C++ binpack loader (compiled at setup via cmake/pybind11)
- **Multi-file round-robin**: 5 files simultaneously for diversity
- **Shuffle buffer**: 512 MB (~5M positions) with reservoir sampling
- **Workers**: 12 CPU workers per GPU, prefetch factor 8
- **RAM usage**: <3 GB (streaming, no full load)
- **Disk cap**: 18 GB max in `/kaggle/working/data/`
- **Resume**: Exact position tracking (file + offset) for checkpoint resume

### 2.3 Download Strategy

| Priority | Source | Method |
|----------|--------|--------|
| 1 | Hugging Face | `hf_hub_download` with resume |
| 2 | HTTP | Range requests with resume |
| 3 | Local | Kaggle dataset mount `/kaggle/input/chess-data` |

- SHA256 checksum verification per file
- Auto-cleanup old files if disk > 20 GB

## 3. Time Budget Math (10 Hours on 2× T4)

| Component | Time Estimate | Notes |
|-----------|---------------|-------|
| Setup (deps, compile, download) | 25 min | Optimized, cached across sessions |
| Data download (18 GB) | 12 min | ~150 MB/s on Kaggle |
| Training (9.5 h) | 570 min | Target |
| Checkpoint/upload overhead | 10 min | 15-min intervals, background |
| Final save + export | 3 min | |
| **Total** | **~10 h** | Fits session limit |

### 3.1 Throughput Targets

- **Batch size**: 32,768 global (16,384 per GPU)
- **Positions/sec/GPU**: ~800k (mixed precision fp16, torch.compile)
- **Effective throughput**: ~1.6M pos/sec (2 GPUs)
- **Positions in 9.5 h**: 9.5 × 3600 × 1.6M ≈ **55 billion**
- **Epochs**: ~3-4 over 18B unique positions (with reshuffling)

### 3.2 Learning Rate Schedule — Cosine with Warm Restarts

```
total_steps = (budget_seconds - setup_seconds) * positions_per_second / batch_size
```

| Phase | Steps | LR Profile |
|-------|-------|------------|
| Warmup | 2% of total | Linear 0 → peak LR |
| Cycle 1 | 50,000 steps | Cosine decay to 0.5% |
| Cycle 2 | 100,000 steps | Cosine decay to 0.5% |
| Cycle 3 | 200,000 steps | Cosine decay to 0.5% |
| ... | (doubling each cycle) | |

**Peak LR**: 0.0012 (tuned for batch size 32,768)  
**Min LR**: 0.5% of peak  
**Warm restarts** help escape local minima and improve final Elo significantly.

## 4. Training Configuration (YAML)

All hyperparameters in `configs/train.yaml`:

```yaml
# Hardware
hardware:
  gpus: 2
  precision: "16-mixed"
  num_workers: 12
  prefetch_factor: 8
  persistent_workers: true
  pin_memory: true
  compile_model: true        # torch.compile max-autotune
  channels_last: true

# Model
model:
  feature_set: "HalfKAv2_hm"
  ft_size: 256
  hidden_sizes: [256, 32]    # L1=256, L2=32
  output_size: 1
  clipped_relu_max: 1.0
  ft_init_scale: 1.0
  hidden_init_scale: 0.5
  output_init_scale: 0.01

# Training
training:
  batch_size: 32768
  max_positions: 50000000000
  learning_rate: 0.0012
  min_lr_ratio: 0.005
  warmup_ratio: 0.02
  weight_decay: 1.5e-4
  gradient_clip_val: 1.0
  wdl_lambda: 0.55
  score_scale: 370.0
  label_smoothing: 0.05
  eval_loss_type: "huber"
  huber_delta: 100.0
  optimizer: "adamw"
  beta1: 0.9
  beta2: 0.999
  eps: 1e-8
  lr_schedule: "cosine_warm_restarts"
  t0: 50000
  t_mult: 2

# Time budget
time_budget:
  hours: 10.0
  setup_minutes: 25
  checkpoint_interval_min: 15
  final_save_min: 3

# Data
data:
  data_dir: "/kaggle/working/data"
  max_data_gb: 18
  shuffle_buffer_mb: 512
  num_files_simultaneous: 5
  drop_last_batch: true
```

## 5. Checkpointing & Resume Strategy

### 5.1 Checkpoint Contents (Complete State)

```python
{
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'scaler': scaler.state_dict(),          # GradScaler for fp16
    'step': global_step,
    'epoch': current_epoch,
    'rng_states': {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    },
    'data_position': dataset.get_state(),   # file_idx + position
    'config': config_dict,
    'wall_time_elapsed': seconds,
}
```

### 5.2 Atomic Save

```python
tmp_path = checkpoint_dir / f"step_{step}.tmp"
torch.save(state, tmp_path)
tmp_path.rename(checkpoint_dir / f"step_{step}.ckpt")  # Atomic on POSIX
```

### 5.3 HF Upload (Background Thread)

- Queue-based uploader (non-blocking)
- Retry: exponential backoff (1s, 2s, 4s, 8s, max 60s)
- Max 3 concurrent uploads
- Never crashes training on failure (logs warning only)
- Uploads: checkpoints, training logs CSV, latest .nnue export

### 5.4 Resume Logic (Exact Restoration)

1. Check `/kaggle/working/checkpoints/` for latest `step_*.ckpt`
2. If none, query HF repo for latest checkpoint
3. Download to local, verify checksum
4. Load **complete state**: model, optimizer, scheduler, scaler, RNG, data position
5. Continue exactly from interrupted step

### 5.5 SIGTERM Handling

- Register signal handler for SIGTERM/SIGINT
- On signal: `should_stop = True`, finish current batch, save final checkpoint, upload, exit(0)

## 6. Export & UCI Usability

### 6.1 Export Process

1. Load best checkpoint (lowest validation loss)
2. Convert to eval mode, fuse Linear+ClippedReLU modules
3. Quantize weights to int8 (per-channel symmetric)
4. Serialize to Stockfish `.nnue` format (exact byte layout)
5. Save to `/kaggle/working/export/nn-<hash>.nnue`

### 6.2 Verification (Automatic)

- Sample 10,000 test positions from held-out data
- Run quantized inference matching Stockfish arithmetic
- Compare PyTorch model output vs. exported net
- Tolerance: ≤0.01 cp MAE, ≤1.0 cp max error
- Fail export if tolerance exceeded

### 6.3 Stockfish Integration

- **Compatible commit**: `stockfish/stockfish@master` (post-NNUE merge)
- **Build script**: `scripts/build_stockfish.sh` (downloads, builds with ARCH=x86-64-vnni)
- **Test script**: `scripts/test_uci.py` (runs `EvalFile`, cutechess match vs baseline)
- **Quick strength check**: 100 games at 10+0.1 vs baseline net

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Kaggle GPU quota exhausted | Medium | High | Detect early, fallback to CPU smoke test |
| Disk full (/kaggle/working 20 GB) | Medium | High | Auto-cleanup, configurable caps, monitor |
| Data download fails | Low | Medium | Multiple sources, resume, local fallback |
| HF upload fails | Medium | Low | Background, retry, never block training |
| OOM on T4 16 GB | Low | High | Batch size auto-tune, gradient accumulation |
| Training diverges | Low | High | Gradient clipping (1.0), LR warmup, Huber loss |
| Session killed before checkpoint | Medium | Medium | 15-min intervals + SIGTERM handler |
| Quantization accuracy loss | Low | Medium | Per-channel quantization, verification step |
| Stockfish version mismatch | Low | Medium | Pin exact commit, document compatibility |
| torch.compile fails | Low | Low | Graceful fallback to eager mode |

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
│   ├── model.py          # HalfKAv2_hm network (optimized)
│   ├── data.py           # Binpack dataset + loader + download
│   ├── train.py          # Training loop (Lightning, optimized)
│   ├── checkpoint.py     # Save/load/resume (atomic)
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
4. Authenticate to HF
5. Detect latest checkpoint (local → HF)
6. Train on both GPUs (DDP, torch.compile, fp16)
7. Checkpoint every 15 min (atomic + HF upload)
8. Export final .nnue + verify
9. Print summary + UCI test instructions
```

### 9.2 Additional Commands

```bash
# Smoke test (2 min)
python run.py --smoke-test

# Benchmark throughput
python run.py --benchmark

# Export only
python run.py --export-only

# Verify export
python run.py --verify-export

# Compile only
python run.py --compile-only

# Download data only
python run.py --download-data
```

### 9.3 Kaggle Notebook (`notebooks/kaggle_train.ipynb`)

3 cells:
1. `!git clone https://github.com/sudo-linux-neko/chess.git && %cd chess`
2. Set `HF_TOKEN` in Kaggle Secrets
3. `!python run.py`

---

*End of DESIGN.md*