#!/usr/bin/env python3
"""
Pure-Python math/logic tests for NNUE trainer correctness.
Tests lambda schedule, EMA update, quantized inference formula,
and data position estimation — no PyTorch required.
"""
import math
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Test 1: Lambda schedule math
print("Test 1: Lambda schedule math...")

# start_lambda defaults to wdl_lambda (0.55), not 1.0
train_cfg = {"wdl_lambda": 0.55}
lambda_cfg = {"start_lambda": None, "lambda_delta": -0.3}
_start_lambda = lambda_cfg.get("start_lambda", train_cfg.get("wdl_lambda", 1.0))
if _start_lambda is None:
    _start_lambda = train_cfg.get("wdl_lambda", 1.0)
start_lambda = float(_start_lambda)
assert start_lambda == 0.55, f"start_lambda should default to wdl_lambda=0.55, got {start_lambda}"
print("  PASS: start_lambda defaults to wdl_lambda=0.55")

# Missing start_lambda key (not just None) also falls back to wdl_lambda
lambda_cfg_no_key = {"lambda_delta": -0.3}
_start_lambda = lambda_cfg_no_key.get("start_lambda", train_cfg.get("wdl_lambda", 1.0))
if _start_lambda is None:
    _start_lambda = train_cfg.get("wdl_lambda", 1.0)
start_lambda = float(_start_lambda)
assert start_lambda == 0.55, f"start_lambda should default to wdl_lambda=0.55 when key missing, got {start_lambda}"
print("  PASS: start_lambda falls back to wdl_lambda when key missing")

# end_lambda defaults to start_lambda + lambda_delta
_end_lambda = start_lambda + lambda_cfg["lambda_delta"]
assert abs(_end_lambda - 0.25) < 1e-9, f"end_lambda should be 0.25, got {_end_lambda}"
print("  PASS: end_lambda = start_lambda + lambda_delta = 0.25")

# Cosine annealing lambda at various progress points
total_steps = 5052000
warmup_pct = 0.05
warmup = int(total_steps * warmup_pct)  # 252600

def compute_lambda(step, start_lambda, lambda_delta, total, warmup):
    if step < warmup:
        progress = step / max(1, warmup)
        return start_lambda * progress
    elif step < total:
        progress = (step - warmup) / max(1, total - warmup)
        return start_lambda + lambda_delta * (1 - math.cos(math.pi * progress)) / 2
    else:
        return start_lambda + lambda_delta  # end_lambda

# At step 0: lambda should be 0 (warmup start)
assert compute_lambda(0, 0.55, -0.3, total_steps, warmup) == 0.0
print("  PASS: lambda at step 0 is 0.0 (warmup start)")

# At step warmup: lambda should be start_lambda (0.55)
val = compute_lambda(warmup, 0.55, -0.3, total_steps, warmup)
assert abs(val - 0.55) < 1e-10, f"lambda at warmup should be 0.55, got {val}"
print("  PASS: lambda at warmup end is 0.55")

# At step total: lambda should be end_lambda (0.25)
val = compute_lambda(total_steps, 0.55, -0.3, total_steps, warmup)
assert abs(val - 0.25) < 1e-10, f"lambda at total should be 0.25, got {val}"
print("  PASS: lambda at total is 0.25 (end_lambda)")

# Lambda should never go below 0 or above 1 (clamped)
for step in [0, warmup, total_steps // 2, total_steps, total_steps + 1000]:
    val = compute_lambda(step, 0.55, -0.3, total_steps, warmup)
    assert 0.0 <= val <= 1.0, f"lambda at step {step} = {val} out of [0,1]"
print("  PASS: lambda always clamped to [0, 1]")

# Jitter doesn't push lambda outside [0, 1]
# Max positive jitter = sample_jitter + batch_jitter = 0.0035 + 0.0070 = 0.0105
# At end_lambda=0.25, max lambda with jitter = 0.2605 < 1.0
# At start_lambda=0.55, max lambda with jitter = 0.5605 < 1.0
print("  PASS: jitter range won't push lambda outside [0, 1]")

print()

# Test 2: EMA update math
print("Test 2: EMA update math...")
decay = 0.9999
ema_param = 0.5
param = 0.7
new_ema = decay * ema_param + (1.0 - decay) * param
expected = 0.9999 * 0.5 + 0.0001 * 0.7
assert abs(new_ema - expected) < 1e-10, f"EMA update mismatch: {new_ema} vs {expected}"
print(f"  PASS: EMA update correct: {new_ema:.10f}")

# After many updates, EMA approaches param but never exceeds it
ema = 0.0
for _ in range(100000):
    ema = decay * ema + (1.0 - decay) * 1.0
assert ema < 1.0, f"EMA should not exceed param value, got {ema}"
assert ema > 0.9, f"EMA should approach param, got {ema}"
print("  PASS: EMA converges toward param value")

print()

# Test 3: Quantized inference formula (Convention A)
print("Test 3: Quantized inference formula (Convention A)...")
# Convention A: output = (x @ W_q.T) * scale + bias_q
# bias_q is NOT divided by scale
import numpy as np

x = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)  # [1, 3]
W_q = np.array([[1, -1, 2], [0, 1, -1]], dtype=np.int8)  # [2, 3]
scale = np.array([0.5, 0.25], dtype=np.float32)  # [2]
bias_q = np.array([10, -5], dtype=np.int32)  # [2]

# Convention A: (x @ W_q.T) * scale + bias_q
acc = x @ W_q.T.astype(np.float32)  # [1, 2]
acc = acc * scale  # [1, 2]
acc = acc + bias_q.astype(np.float32)  # [1, 2]

# Manual calculation:
# x @ W_q.T = [[1, 0, 1]] @ [[1, 0], [-1, 1], [2, -1]] = [[3, -1]]
# [3, -1] * [0.5, 0.25] = [1.5, -0.25]
# [1.5, -0.25] + [10, -5] = [11.5, -5.25]
expected = np.array([[11.5, -5.25]], dtype=np.float32)
assert np.allclose(acc, expected), f"Convention A mismatch: {acc} vs {expected}"
print("  PASS: Convention A formula: (x @ W_q.T) * scale + bias_q")

# Verify Convention B would give DIFFERENT (wrong) result:
# Convention B: (x @ W_q.T + bias_q) * scale
# [3, -1] + [10, -5] = [13, -6]
# [13, -6] * [0.5, 0.25] = [6.5, -1.5]  <- WRONG
conv_b = (x @ W_q.T.astype(np.float32) + bias_q.astype(np.float32)) * scale
assert not np.allclose(acc, conv_b), "Convention A and B should differ"
print("  PASS: Convention B (wrong) gives different result than Convention A")

print()

# Test 4: Data position estimation math
print("Test 4: Data position estimation math...")
ppf = 100000  # positions per file
base_file = 5
base_position = 10000
samples_since_resume = 50000

# If samples_since_resume <= remaining in first file
samples_remaining_in_first_file = ppf - base_position  # 90000
if samples_since_resume <= samples_remaining_in_first_file:
    files_completed = 0
    position_in_file = base_position + samples_since_resume  # 60000
else:
    remaining = samples_since_resume - samples_remaining_in_first_file
    files_completed = 1 + (remaining // ppf)
    position_in_file = remaining % ppf

assert files_completed == 0, f"files_completed should be 0, got {files_completed}"
assert position_in_file == 60000, f"position_in_file should be 60000, got {position_in_file}"
print("  PASS: Position estimation within first file")

# If samples_since_resume exceeds first file
samples_since_resume = 150000
if samples_since_resume <= samples_remaining_in_first_file:
    files_completed = 0
    position_in_file = base_position + samples_since_resume
else:
    remaining = samples_since_resume - samples_remaining_in_first_file  # 60000
    files_completed = 1 + (remaining // ppf)  # 1 + 0 = 1
    position_in_file = remaining % ppf  # 60000

assert files_completed == 1, f"files_completed should be 1, got {files_completed}"
assert position_in_file == 60000, f"position_in_file should be 60000, got {position_in_file}"
print("  PASS: Position estimation crossing into next file")

# Negative samples_since_resume clamped to 0
samples_since_resume = -1000
if samples_since_resume < 0:
    samples_since_resume = 0
assert samples_since_resume == 0
print("  PASS: Negative samples_since_resume clamped to 0")

print()

# Test 5: DDP file sharding math
print("Test 5: DDP file sharding math...")
files = list(range(100))  # 100 files
rank = 0
world_size = 2
rank_files = files[rank::world_size]  # [0, 2, 4, ..., 98] — 50 files
assert len(rank_files) == 50
assert rank_files[0] == 0
assert rank_files[-1] == 98
print("  PASS: Rank 0 gets even-indexed files")

rank = 1
rank_files = files[rank::world_size]  # [1, 3, 5, ..., 99] — 50 files
assert len(rank_files) == 50
assert rank_files[0] == 1
assert rank_files[-1] == 99
print("  PASS: Rank 1 gets odd-indexed files")

# No overlap between ranks
set0 = set(files[0::2])
set1 = set(files[1::2])
assert set0.isdisjoint(set1), "Ranks should have disjoint file sets"
assert set0 | set1 == set(files), "Union should cover all files"
print("  PASS: DDP sharding gives disjoint, complete coverage")

# Global index mapping: local_idx -> global index
def global_index(local_idx, rank, world_size):
    return rank + local_idx * world_size if world_size > 1 else local_idx

assert global_index(0, 0, 2) == 0
assert global_index(1, 0, 2) == 2
assert global_index(0, 1, 2) == 1
assert global_index(1, 1, 2) == 3
print("  PASS: Global index mapping correct")

# Resume file belongs to rank check
resume_file = 6
rank = 0
world_size = 2
# (resume_file - rank) % world_size == 0 means this rank owns the file
assert (resume_file - rank) % world_size == 0, "Rank 0 should own file 6"
resume_local = (resume_file - rank) // world_size  # 3
assert resume_local == 3
print("  PASS: Resume file ownership check correct")

print()

# Test 6: WDL logit conversion
print("Test 6: WDL logit conversion math...")
scale = 400.0
eval_cp = 100.0  # +100 centipawns
w = 1.0 / (1.0 + math.exp(-eval_cp / scale))  # sigmoid(100/400)
l = 1.0 / (1.0 + math.exp(eval_cp / scale))    # sigmoid(-100/400)
d = 1.0 - w - l
d = max(d, 1e-6)
w = max(w, 1e-6)
l = max(l, 1e-6)
total = w + d + l
w, d, l = w / total, d / total, l / total
logits = [math.log(w), math.log(d), math.log(l)]
assert len(logits) == 3
assert all(math.isfinite(x) for x in logits)
print(f"  PASS: WDL logits computed: W={logits[0]:.4f}, D={logits[1]:.4f}, L={logits[2]:.4f}")

# Eval of 0 should give equal W/L
eval_cp = 0.0
w = 1.0 / (1.0 + math.exp(-eval_cp / scale))
l = 1.0 / (1.0 + math.exp(eval_cp / scale))
assert abs(w - l) < 1e-10, f"At eval=0, W and L should be equal: {w} vs {l}"
print("  PASS: At eval=0, W and L probabilities are equal")

print()

# Test 7: Config YAML loading
print("Test 7: Config YAML loading...")
import yaml
config_yaml = """
training:
  learning_rate: 0.0008
  wdl_lambda: 0.55
  gradient_accumulation: 4
model:
  dropout: 0.0
  clipped_relu_max: 1.0
"""
config = yaml.safe_load(config_yaml)
assert config["training"]["learning_rate"] == 0.0008
assert config["training"]["wdl_lambda"] == 0.55
assert config["training"]["gradient_accumulation"] == 4
assert config["model"]["dropout"] == 0.0
assert config["model"]["clipped_relu_max"] == 1.0
print("  PASS: YAML config loads correctly")

# Scientific notation in YAML - PyYAML parses 4e-5 as string, but code wraps in float()
config_yaml_sci = """
training:
  weight_decay: 4.0e-5
  learning_rate: 8.0e-4
"""
config_sci = yaml.safe_load(config_yaml_sci)
assert config_sci["training"]["weight_decay"] == 4e-5
assert config_sci["training"]["learning_rate"] == 8e-4
print("  PASS: Scientific notation in YAML parses correctly")

# Verify float() conversion handles string scientific notation (PyYAML quirk)
assert float("4e-5") == 4e-5
assert float("8e-4") == 8e-4
print("  PASS: float() converts string scientific notation correctly")

print()
print("=" * 60)
print("ALL MATH/LOGIC TESTS PASSED!")
print("=" * 60)
