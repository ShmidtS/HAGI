# HAGI Training Stack

HAGI's training stack is built for a single RTX 3070 (8GB VRAM) running a ~74M parameter model with 3B tokens. Every component is constrained by the 8GB budget.

For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md). For config reference, see [CONFIGURATION.md](CONFIGURATION.md).

---

## The Stack

| Layer | Component | Why |
|-------|-----------|-----|
| Recipe | Smol Training Playbook | Configs/hyperparams/data-mix for this exact scale |
| Tokenizer | SmolLM2 (~49K vocab) | Trained on edu+code+math; right vocab size for 74M (embedding ~33% of params) |
| Data processing | `datatrove` | Tokenize + shard FineWeb-Edu / code / math into .bin streams |
| Data loading | `hagi.data` | nanoGPT-style memmap `get_batch`, sequential cycling, prefix-LM, SFT |
| Training loop | `scripts/train.py` | 1038-line canonical script: VRAM probe, distillation, full resume |
| Core loop | `hagi.train.loop` | Single source of truth for fwd/bwd/step: EMA, NARS, NaN guard, timing |
| Optimizer | Muon+AdamW (`hagi.train.optim`) | Muon for 2D weights (Newton-Schulz), AdamW for embeddings/norms/gates |
| Distillation | `hagi.train.distillation` | SmolLM2-135M teacher: embedding transfer + KL divergence |
| Evaluation | `hagi.eval` | lm-eval-harness adapter, golden benchmarks, intelligence-density metrics |
| RL training | `scripts/train_rl.py` | MGPO loop (on-policy, group-relative advantage) |

---

## Canonical Training Command

```bash
python scripts/train.py --config configs/rtx3070_canonical.yaml
```

### `scripts/train.py` vs `hagi-train`

`scripts/train.py` is the **canonical** training path (1038 lines): sequential cycling, VRAM probe, prefix LM, distillation integration, full resume. `hagi-train` (CLI) is a thin dispatcher with fewer features. **Use the script for real training.**

---

## Workflow

### 1. Data Preparation

```bash
# Needs HF_TOKEN in .env
python scripts/download_data.py
```

Downloads and tokenizes datasets into `data/v4_3b/*.bin` (memmap-packed, uint16). Sources:

| Source | Weight | Phase |
|--------|--------|-------|
| tinystories | 0.02 | 1 (easy) |
| python_instruct | 0.02 | 1 (easy) |
| smoltalk | 0.03 | 1 (easy) |
| wikipedia_en | 0.10 | 2 (mid) |
| wikipedia_ru | 0.08 | 2 (mid) |
| openwebmath | 0.03 | 2 (mid) |
| oscar_ru | 0.07 | 3 (hard) |
| slimpajama | 0.25 | 3 (hard) |
| edu | 0.40 | 3 (hard) |

### 2. Dry Run (VRAM Check)

```bash
python scripts/train.py --dry-run
```

Builds the model, runs one batch, reports VRAM usage, and exits. Use this to verify the config fits your GPU before starting a real run.

### 3. Train

```bash
# Full training
python scripts/train.py --config configs/rtx3070_canonical.yaml

# Resume from checkpoint
python scripts/train.py --resume checkpoints/rtx3070/step-00039000.pt

# With torch.compile on Windows
torch-compile.bat scripts/train.py --config configs/rtx3070_canonical.yaml
```

### 4. Evaluate

```bash
# Via hagi-eval CLI
hagi-eval --checkpoint checkpoints/rtx3070/step-00039000.pt --golden

# Via Python module
python -m hagi.eval.evaluate --ckpt checkpoints/rtx3070/step-00039000.pt --benchmarks gsm8k,arc_challenge
```

---

## Optimizer: Muon + AdamW

### Architecture

The optimizer is a **hybrid**: Muon for 2D weight matrices, AdamW for everything else.

| Component | Optimizer | Why |
|-----------|-----------|-----|
| 2D hidden weights (attention, MLP, GDR-MLP) | Muon | Orthogonalized updates via Newton-Schulz; scale-invariant, faster convergence |
| Embeddings, LM head, norms, gates, iteration embeddings | AdamW | Standard optimizer for non-matrix params |

### Muon Details

- **Algorithm**: Quintic Newton-Schulz iteration (5 steps by default) approximates `G(G^T G)^{-1/2}` to orthogonalize the gradient.
- **LR**: 0.02 (canonical Keller Jordan recipe). Scale-invariant: higher LR converges faster without grad clipping.
- **Momentum**: 0.97 (Nesterov-style EMA). Longer momentum smooths per-step update direction across doubled pass count (L_cycles=2).
- **Weight decay**: 0.5 (scale-aware). `wd=0.5 -> ||W||_ss ~ 2.0`, matching the residual-scaled init. This **structurally bounds** the 2D hidden weight norms that grew unbounded under Muon without decay, which caused the residual-stream divergence.
- **NS steps**: 5 (canonical). 10 adds ~50% opt time for marginal quality gain.

### AdamW Details

- **LR**: 0.0004 (cosine/WSD schedule)
- **Betas**: (0.9, 0.98). beta2 0.98 smooths variance EMA, damping grad scale spikes.
- **Weight decay**: 0.1 (decoupled, AdamW style)
- **Eps**: 1e-8

### Schedule-Free AdamW

Alternative optimizer available via `optimizer: schedule-free-adamw`. No LR schedule needed; the optimizer self-adapts.

Source: `src/hagi/train/optim.py`

---

## Precision Strategy

### manual_bf16 (Default)

```
precision: manual_bf16
```

- Model cast to bf16 at construction
- No autocast (no autocast dispatch overhead)
- Grads accumulated in fp32 inside the loop
- NOT bf16-autocast: full bf16-autocast with fp32 master weights blew VRAM to 10GB and ran 6.7x slower on 8GB RTX 3070

### Mixed-Precision Optimizations

| Feature | Config | Effect |
|---------|--------|--------|
| fp16 attention | `fp16_attention: true` | QKV cast to fp16 for SDPA softmax (10 mantissa bits vs bf16's 7) |
| fp32 RMSNorm | `fp32_rmsnorm: true` | Upcast to fp32 for variance computation (exact mean(x^2)) |
| fp32 grad accum | `fp32_grad_accum: false` | DISABLED: fused AdamW requires matching dtype for params+grads+moments |
| INT8 KV cache | `int8_kv_cache: true` | Inference-only: 2x cache memory reduction |

---

## Knowledge Distillation

### Teacher

SmolLM2-135M (or SmolLM2-360M for distillation phase). Provides:
1. **Pretrained token embeddings** (exact copy at init)
2. **Soft logit targets** for KL divergence during training

### Student Loss

```
L_student = alpha * CE_hard + (1 - alpha) * T^2 * KL(soft_student || soft_teacher)
```

### VRAM Strategy

Teacher forward returns **hidden states** (14MB), NOT logits (1.2GB). Both student and teacher hidden are projected to logits per-chunk inside KL, so peak logits memory = `2 * chunk_size * V * dtype_bytes`, never `[B, T, V]`.

### Teacher Lifecycle

Teacher is freed at 60% of training (`step 87891` for 146K total steps), reclaiming ~722MB VRAM for larger batch or longer training.

Source: `src/hagi/train/distillation.py`

---

## Data Pipeline

### Sequential Cycling Curriculum

`sequential_cycles: 3` iterates datasets in difficulty order 3 times before shuffling:

```
Phase 1 (easy): tinystories -> python_instruct -> smoltalk
Phase 2 (mid):  wikipedia_en -> wikipedia_ru -> openwebmath
Phase 3 (hard): oscar_ru -> slimpajama -> edu
```

Each phase is a full epoch over its datasets. After 3 cycles, all datasets are shuffled together for the remaining steps.

### Memmap Packed Format

- Binary memmap files (`data/v4_3b/*.bin`), uint16 dtype
- Best-Fit-Decreasing packing on EOS boundaries
- `MemmapDataset` + `SequentialCyclingIterator` handle iteration
- `PrefixLMBatch` creates prefix-LM masks (bidirectional prefix, causal suffix)

### Variable-Length Training

- `min_seq_len` / `max_seq_len`: each sample draws a random length in this range
- Fixed at 1024/1024: every position carries gradient (densest signal/step)
- Single T kills the compile recompile-spikes that variable-length triggers

Source: `src/hagi/data/`

---

## Composite Loss

```
L_total = L_CE
        + w_iso    * L_iso         (0.02, HDIM invariant alignment)
        + w_moe    * L_moe         (0.005, expert load-balance)
        + w_msa_lb * L_msa_lb      (0.01, MSA router load-balance)
        + w_gdr_router * L_gdr     (0.005, GDR capacity router load-balance)
```

### Warmup

All auxiliary losses ramp from 0 to target over `warmup_steps` (1000 steps):
- `w_aux_start: 0.0`, `w_iso_start: 0.0`, `w_moe_start: 0.0`, `w_msa_lb_start: 0.0`, `w_gdr_router_start: 0.0`

### Fused CE

When `use_fused_ce=true`:
- `logits` is `None` in training forward
- Loss computed via `fused_linear_cross_entropy` (no `[B, T, V]` materialization)
- Peak memory: `ce_fused_chunk_size * V * dtype_bytes` (e.g. `4096 * 49152 * 2B = 384MB`)
- Label smoothing: 0.05

### Disabled Losses

- `L_aux` (contrastive): DISABLED (w_aux=0.0). Labels were the batch index -- trivially true, contradicted the LM objective.
- `L_quality` (quality head BCE): DISABLED (w_quality=0.0). Head never fires when `use_fused_ce=true`.
- `magic_norm_max`: DISABLED (0.0). Per-blade gradient norm clip is a crutch; weight norms are bounded structurally by Muon weight decay.

Source: `src/hagi/losses.py`, `src/hagi/train/loop.py`

---

## LR Schedule: WSD

```
schedule: wsd
```

**W**armup-**S**table-**D**ecay: better for small models with embedding transfer.

| Phase | Steps | LR |
|-------|-------|-----|
| Warmup | 0 -- 1000 | 0 -> 0.0004 (linear) |
| Stable | 1000 -- 131836 | 0.0004 (peak) |
| Decay | 131836 -- 146485 | 0.0004 -> 0.00004 (linear, last 10%) |

Warmup is short (1000 steps) because embeddings are pretrained -- no random-init chaos.

---

## Checkpoints

### Save

`save_checkpoint(model, optimizer, step, ckpt_dir)` writes `step-<N>.pt` into `checkpoints/rtx3070/`. The model config is stored as a plain dict (not a pickled dataclass), so checkpoints load under torch's default `weights_only=True` -- no arbitrary code execution on load.

### Resume

```bash
python scripts/train.py --resume checkpoints/rtx3070/step-00039000.pt
```

Restores: model state_dict, optimizer state, step counter, LoopConfig. The canonical `train.py` handles the full resume path (data position, scheduler, EMA).

### Interval

`ckpt_interval: 500` -- checkpoint every 500 steps.

Source: `src/hagi/train/checkpoint.py`

---

## RL Training (MGPO)

```bash
python scripts/train_rl.py --config configs/rl_rtx3070.yaml
```

### MGPO (MaxEnt-Guided Policy Optimization)

Adapted from VibeThinker-3B for single-GPU 8GB:
- Group size G=4 (4 rollouts per prompt)
- On-policy: generate -> reward -> update (no replay buffer)
- Gradient checkpointing during update phase
- Sequential rollout (one prompt at a time) to bound VRAM

### Key Differences from GRPO

- **MGPO prompt weight**: `w(q) = exp(-gamma * |p(q) - p0|)` focuses updates on prompts near the model's capability boundary (p ~ 0.5)
- **Long2Short reward shift**: zero-sum brevity redistribution

Source: `src/hagi/train/rl_loop.py`, `src/hagi/train/rewards.py`

---

## torch.compile on Windows

`torch-compile.bat` wraps `python` with MSVC `vcvars64.bat` for Triton/Inductor:

```bash
torch-compile.bat scripts/train.py --config configs/rtx3070_canonical.yaml
```

Use this instead of bare `python` when `compile: true` in config.

---

## Hardware Notes

- ~74M model trains on RTX 3070 8GB in manual_bf16 with gradient checkpointing.
- `batch_size=10` is VRAM-optimal: batch=11 -> 7.42GB reserved (over 7GB budget).
- Teacher model (SmolLM2-360M, 722MB bf16) loaded during distillation phase; freed at 60% training.
- Gradient checkpointing required: without GC, teacher + fp32 grads + activations -> OOM.
- Group checkpointing size 2: wraps 2 consecutive blocks in one checkpoint call (balanced [2,2,2,1] split for 7-block reasoning core).
- CPU is fine for dry-run only.

---

## Profiling

```bash
python scripts/profile_steps.py --config configs/rtx3070_canonical.yaml
```

Synthetic GPU-resident tokens, isolates compute from I/O. Reports per-step timing breakdown.

---

## EMA

Exponential Moving Average of model parameters, maintained alongside the main model. EMA weights are saved in checkpoints and used for evaluation. Controlled by `ema_decay` in LoopConfig.

---

## Training Loop Invariants

- **CE-computed-once contract**: cross-entropy is computed exactly once per step, never twice (even with composite loss warmup).
- **NaN guard**: if loss is NaN, the step is skipped (no optimizer update) and a warning is logged.
- **Component logging**: per-component loss values, gradient norms, timing breakdown, and EMA metrics are logged every step.
- **Single source of truth**: `loop.py` is the only place forward/backward/step happens. `train.py` builds the data source and calls `train()`.

Source: `src/hagi/train/loop.py`
