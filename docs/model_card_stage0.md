---
license: apache-2.0
language:
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- hagi
- grade-decomposed-recurrence
- clifford-algebra
- small-language-model
- research
- from-scratch
datasets:
- HuggingFaceFW/fineweb-edu
---

# HAGI — Stage 0: Model A (Dense Baseline)

A **113M-parameter** dense transformer trained **from scratch** on **1B tokens** of
FineWeb-Edu. This is the **control baseline** for the HAGI research project — the
reference point against which the project's novel mechanism, **Grade-Decomposed
Recurrence (GDR)**, will be measured.

> ⚠️ **Research artifact, not a product.** It is intentionally small and
> undertrained (1B tokens, ~0.4× Chinchilla-optimal for this size). It writes
> fluent, on-topic English but its **facts are unreliable**. Its purpose is
> scientific comparison, not deployment.

## What is HAGI?

HAGI (Hypercomplex Artificial General Intelligence) investigates a single
hypothesis: *does decomposing a recurrent transformer's hidden state into
**Clifford-algebra grades** — with per-grade update dynamics and geometric-product
cross-grade interaction — improve reasoning per parameter in small models?*

Standard recurrent-depth transformers iterate over a flat hidden vector; gains
plateau after a few iterations because every dimension converges at the same rate.
HAGI splits the state into scalar / vector / bivector / trivector grades that
evolve at different rates, so each reasoning iteration has different dynamics.

This checkpoint is **Model A**: the plain dense baseline with **no recurrence and
no Clifford structure** — the experimental control.

## This checkpoint

| | |
|---|---|
| Role | **Model A** — dense baseline (the control) |
| Recurrence (`use_loop`) | ❌ none |
| Clifford GDR (`use_gdr`) | ❌ none |
| Parameters | 113.3M |
| File | `step-00003815.pt` (PyTorch; model + optimizer state + config dict) |
| Final train loss | **~3.43** (perplexity ≈ 31) |
| Status | Stage 0 baseline — trained, stable, validated |

## Architecture

A **Perception → Reasoning → Expression** transformer. For Model A the reasoning
core is a plain stack (no looping, no grade decomposition).

| Component | Value |
|-----------|-------|
| Hidden size | 768 |
| Layers | 12 (4 perception / 4 reasoning / 4 expression) |
| Attention | Grouped-Query Attention (12 query heads, 4 KV heads) |
| MLP | SwiGLU (768 → 2048 → 768) |
| Positional | RoPE (θ=10000) |
| Norm | RMSNorm (pre-norm) |
| Embeddings | weight-tied input/output |
| Vocabulary | 49,152 (SmolLM2 tokenizer) |
| Context length | 4096 |
| Precision | bf16 |

## Training

| | |
|---|---|
| Data | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample-10BT`, ~630M unique tokens |
| Tokens seen | 1.0B (~1.6 epochs) |
| Sequence length | 4096 |
| Optimizer | AdamW (lr 3e-4, wd 0.1, cosine decay, 400-step warmup) |
| Effective batch | 262,144 tokens/step (batch 8 × grad-accum 8 × seq 4096) |
| Steps | 3,815 |
| Loss objective | next-token cross-entropy (fp32-accumulated, chunked) |
| Hardware | Google Colab **A100-40GB** (bf16 + FlashAttention-2 + `torch.compile`) |
| Resumption | checkpoints mirrored to this repo; the run survived an A100→L4→A100 platform switch via HF-Hub resume |

Training was dead stable — loss fell **10.95 → 3.43** with no spikes or
divergence, clean cosine decay to the LR floor.

## Sample generation

Prompt: *"The sun is a star that"* (temperature 0.8):

> *"The sun is a star that is standing in the solar system. The next planet is the
> moon, and when a planet hits the sun it can take a few minutes to drop it into
> space. In order to get a close look, astronomers have used a technique called
> alternating ablation, in which the plasma rises..."*

Grammatical, coherent, educational register — the language modeling clearly works.
Facts are wrong, as expected at this scale.

## How to load

This is a raw PyTorch checkpoint (not a `transformers` model). Load it with the
HAGI prototype code:

```bash
git clone -b experimental https://github.com/ShmidtS/HAGI.git && cd HAGI
pip install -r requirements.txt
huggingface-cli download NAME0x0/hagi-stage0 step-00003815.pt --local-dir checkpoints/stage0_a100
```

```python
import torch, torch.nn.functional as F
from prototype.training.loop import load_checkpoint
from prototype.data.tokenizer import load_tokenizer

model, step = load_checkpoint("checkpoints/stage0_a100/step-00003815.pt", device="cuda")
model.eval()
tok = load_tokenizer("HuggingFaceTB/SmolLM2-135M")

x = torch.tensor([tok.encode("The sun is a star that")], device="cuda")
for _ in range(60):
    with torch.no_grad():
        logits = model(x)[0, -1]
    nxt = torch.multinomial(F.softmax(logits / 0.8, dim=-1), 1)
    x = torch.cat([x, nxt.view(1, 1)], dim=1)
print(tok.decode(x[0].tolist()))
```

Inference fits comfortably in <1GB VRAM (or CPU) — it is far cheaper than training.

## The ablation (where this fits)

Model A is one of four models trained identically (same data, schedule,
tokenizer) that differ only in architecture:

| Model | Recurrence | Clifford GDR | Tests |
|-------|-----------|--------------|-------|
| **A (this)** | ❌ | ❌ | dense baseline (control) |
| B | ✅ loop ×3 | ❌ | recurrence only |
| C | ❌ | ✅ | Clifford bolted on |
| D | ✅ loop ×3 | ✅ | full GDR |

The decisive comparison is **B vs D**: same parameters and compute pattern, the
only difference being grade decomposition. A positive result there is the
publishable contribution.

## Limitations

- **Scale:** 113M parameters — far below any production model.
- **Undertrained:** 1B tokens (~0.4× Chinchilla-optimal); a stronger baseline
  needs ~2.3B tokens on more unique data.
- **Context:** trained at 4096 but on educational web text only.
- **Reliability:** generates plausible but frequently incorrect statements. Do not
  use for factual or decision-making purposes.
- **Scope:** an experimental control, released for reproducibility of the HAGI
  ablation — not optimized for any downstream task.

## Links

- **Code:** https://github.com/ShmidtS/HAGI (branch `experimental`)
- **Architecture spec:** see `docs/ARCHITECTURE.md` in the repo
- **Milestones / research plan:** see `docs/MILESTONES.md`

## Citation

```bibtex
@software{hagi2026,
  title  = {HAGI: Grade-Decomposed Clifford Recurrence for Intelligence-Dense Small Models},
  author = {HAGI Contributors},
  url    = {https://github.com/ShmidtS/HAGI},
  year   = {2026}
}
```

## License

Apache 2.0.
