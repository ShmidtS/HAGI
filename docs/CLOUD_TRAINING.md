# Free & Cheap Cloud Training for HAGI

Practical guide to training HAGI on free/low-cost cloud GPUs, and the kernel/
memory optimizations the prototype ships to make those runs faster. For the
training stack itself see [TRAINING.md](TRAINING.md); for hardware sizing see its
Hardware Notes.

> **Honest bottom line.** The local dev box (4.3GB) cannot train. Free cloud GPUs
> (16GB T4/P100) *can* run a **proof-of-life** — validate the pipeline and get a
> real loss curve — but a full Stage 0 (5B tokens) is ~weeks of fragmented
> sessions. Use free tiers to de-risk; rent an Ampere GPU for the canonical run.

---

## Free GPU platforms (June 2026)

| Platform | GPU(s) | VRAM | Session cap | Quota | Persistent storage | Best for |
|----------|--------|------|-------------|-------|--------------------|----------|
| **Kaggle** | P100 *or* T4×2 | 16GB (×2 for T4) | 12h | ~30h/week | No (use Kaggle Datasets) | Longest free weekly hours |
| **Google Colab (free)** | T4 | 16GB | 12h, 90min idle | ~15–30h/week, dynamic | No (mount Drive) | Quick iteration |
| **Lightning AI** | L4 / L40S / A100 | 24–80GB | studio-based | ~80 GPU·h/month, no CC | **50GB persistent** | **Best fit — Ampere+, bf16, FA-2, Muon all work** |
| **SageMaker Studio Lab** | (entry GPU) | ~16GB | 4h/session | daily | 15GB persistent | No-credit-card simplicity |
| **Paperspace Gradient (free)** | M4000/T4-class | 8–16GB | ~6h | when available | 5–10GB | Backup option |

Verify current quotas before relying on them — free tiers change often.

### Why Lightning AI is the standout for HAGI

It is the only free tier listed with **Ampere+ GPUs (sm80+)** and persistent
storage. That matters concretely:

- **bf16 tensor cores** → the canonical `precision: bf16` configs run as-is.
- **FlashAttention-2** (needs sm80+) is available via SDPA's flash backend.
- **Muon** (needs bf16/fp32) runs at speed → the optimizer ablation is possible.
- **50GB persistent storage** → checkpoints survive, resume is trivial.

On Kaggle/Colab T4 none of the above hold (see precision notes below).

---

## GPU-specific gotchas

### Precision is per-architecture — do not blindly use bf16

| GPU | Arch | Fast precision | Notes |
|-----|------|----------------|-------|
| T4 | Turing (sm75) | **fp16** | No bf16 tensor cores. bf16 runs but slow. |
| P100 | Pascal (sm60) | **fp32** | No fp16 *tensor cores* either; fp16 is weak. |
| L4 / A100 / L40S | Ampere+ (sm80+) | **bf16** | Full bf16 + FlashAttention-2. |

- `configs/colab_t4.yaml` already sets `precision: fp16` for this reason.
- **Muon needs bf16 or fp32** (the loss-scaler path doesn't support fp16+Muon).
  On a T4 that means Muon → fp32 (slow). **Defer the Muon ablation to an Ampere
  GPU**; keep `optimizer: adamw` on free T4 runs.

### FlashAttention-2 does NOT run on T4

FA-2 requires sm80+ (Ampere). T4 is sm75. HAGI does not depend on the `flash-attn`
package — attention is `torch.nn.functional.scaled_dot_product_attention`
(`prototype/model/transformer.py`), which **auto-dispatches** to the best
available backend: FlashAttention-2 on Ampere+, the memory-efficient kernel on T4.
Nothing to install; nothing to change per GPU.

### Sessions die — checkpoint often and resume

Free sessions are killed at 12h (and ~90min idle on Colab). Training MUST be
resumable or a multi-day run is lost:

```bash
# First launch
python -m prototype.training.train --config configs/colab_t4.yaml --data data/shards

# After a session is killed — resume the latest checkpoint (weights + optimizer)
python -m prototype.training.train --config configs/colab_t4.yaml --data data/shards --resume auto
```

`--resume auto` loads the newest `checkpoints/<name>/step-*.pt` and restores model
**and optimizer** state, continuing the LR schedule from the saved step. Keep
`ckpt_interval` small (the free-tier config uses 500).

### No persistent storage on Kaggle/Colab

- **Colab:** mount Google Drive and point `ckpt_dir` (and shards) at it, or the
  12h reset wipes everything.
- **Kaggle:** write checkpoints to `/kaggle/working` and snapshot to a Kaggle
  Dataset between sessions; shards live in an input Dataset.
- **Lightning AI / Studio Lab:** persistent storage — no extra step.

---

## The acceleration stack (what HAGI actually uses)

These are wired into the prototype; toggle via config. They are the real speed/
memory levers.

| Lever | Where | Effect | Default |
|-------|-------|--------|---------|
| **SDPA / FlashAttention** | `transformer.py` (always on) | O(N) attention memory; FA-2 on Ampere | always |
| **`torch.compile`** | `training.compile: true` | Fused **Triton** kernels, ~1.3–2× | off (on for `colab_t4`) |
| **Gradient checkpointing** | `model.gradient_checkpointing: true` | ~30% recompute for big activation-memory cut | off (on for `colab_t4`) |
| **Chunked cross-entropy** | `model.ce_chunk_size: N` | Avoids the ~13GB fp32 logit spike at full seq | 4096 (canonical), 1024 (T4) |
| **TF32 matmul** | auto (`train.py`) | Free ~1.3–2× fp32 on Ampere | always |
| **bf16/fp16 AMP** | `training.precision` | Halve activation memory + tensor cores | bf16 / fp16 (T4) |

> **`torch.compile` is "using Triton."** PyTorch's compiler lowers the model to
> fused Triton kernels automatically. Hand-writing custom Triton kernels is a
> Stage-5 concern (the Clifford geometric product is the one candidate worth a
> custom kernel) — premature before a baseline trains.

---

## Notebook launcher (Colab / Kaggle / Lightning Jupyter)

`notebooks/hagi_cloud_train.ipynb` runs the whole flow cell-by-cell: clone →
install → **auto-detect GPU and pick the config** (bf16 `baseline` on Ampere+,
fp16 `colab_t4` on a T4) → mount persistent storage → tokenize → train with
`--resume auto`. Easiest path on Colab (mounts Drive automatically). For a
terminal (Lightning AI), the `scripts/cloud_*.sh` below are equivalent.

## Recommended workflow

1. **Tokenize once, store as a Dataset** (Kaggle Dataset / Drive folder /
   Lightning storage). Tokenization is CPU-bound — do it once, reuse the shards.
2. **Proof-of-life on free tier** with `configs/colab_t4.yaml` (~500M tokens,
   ~15k steps). Goal: loss decreases smoothly, no NaNs, checkpoints + `--resume`
   work across a session kill. This validates the whole pipeline for $0.
3. **Canonical Stage 0 on Ampere.** Either Lightning AI (free, if quota allows) or
   a rented 24GB GPU (4090/A5000/L4, ~\$0.3–0.5/h). Use `configs/baseline.yaml`
   (bf16, full 4096 seq, 5B tokens derived to ~19k steps). Enable
   `compile: true`; enable `gradient_checkpointing: true` only if memory is tight.
4. **Ablation (B/C/D) + Muon** on the Ampere box, holding tokens/step constant.

---

## Sources

- [Kaggle — Efficient GPU Usage](https://www.kaggle.com/docs/efficient-gpu-usage), [Weekly GPU quota](https://www.kaggle.com/general/108481)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html), [Colab free-tier T4 guide (2026)](https://aicreditmart.com/ai-credits-providers/google-colab-free-tier-t4-gpu-access-guide-2026/)
- [Free cloud GPU comparison (2026)](https://aimultiple.com/free-cloud-gpu), [GPU Tracker — Colab alternatives](https://gputracker.dev/blog/google-colab-alternatives)
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) (sm80+ requirement), [Turing support issue #720](https://github.com/Dao-AILab/flash-attention/issues/720)
