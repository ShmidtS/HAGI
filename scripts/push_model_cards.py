"""Generate and push a README.md (model card) to each HF ablation repo.

Each card explains HAGI + GDR, marks this model's exact role in the four-way
ablation, lists the shared architecture/training setup, shows how to load it, and
links its three siblings. Re-runnable (overwrites the README each time).

    # preview one card without uploading
    python scripts/push_model_cards.py --user NAME0x0 --dry-run

    # push README.md to all four repos (needs a write token in HF_TOKEN)
    python scripts/push_model_cards.py --user NAME0x0
"""

from __future__ import annotations

import argparse
import io

# Per-model facts. The two flags (loop / gdr) are the ONLY differences.
MODELS = {
    "a": dict(name="Model A - Dense Baseline (control)", loop=False, gdr=False,
              params="113.3M", role="dense baseline - the experimental control / floor"),
    "b": dict(name="Model B - Recurrence Only", loop=True, gdr=False,
              params="113.3M", role="recurrence alone (looped reasoning core, flat hidden state)"),
    "c": dict(name="Model C - Clifford, Bolted-On", loop=False, gdr=True,
              params="114.6M", role="Clifford grade decomposition WITHOUT recurrence"),
    "d": dict(name="Model D - Full GDR", loop=True, gdr=True,
              params="114.6M", role="loop + grade decomposition (the full mechanism under test)"),
}
ORDER = ["a", "b", "c", "d"]


def _ablation_table(this: str) -> str:
    rows = []
    for k in ORDER:
        loop = "loop x3" if MODELS[k]["loop"] else "-"
        gdr = "grades" if MODELS[k]["gdr"] else "-"
        label = f"**{k.upper()} (this model)**" if k == this else k.upper()
        rows.append(f"| {label} | {loop} | {gdr} | {MODELS[k]['params']} |")
    return "\n".join(rows)


def card(m: str, user: str) -> str:
    f = MODELS[m]
    loop_txt = "loop x3" if f["loop"] else "none"
    gdr_txt = "grades (scalar/vector/bivector/trivector + geometric product)" if f["gdr"] else "none"
    return f"""---
license: apache-2.0
language:
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- hagi
- grade-decomposed-recurrence
- clifford-algebra
- ablation
- small-language-model
- research
datasets:
- HuggingFaceFW/fineweb-edu
---

# HAGI Ablation - {f['name']}

One of **four same-budget models** in the HAGI **Grade-Decomposed Recurrence (GDR)**
ablation. All four share data, tokenizer, schedule, and token budget and differ in
**only** two flags - `use_loop` and `use_gdr` - so the comparison isolates the
mechanism with no confounds.

> **Research artifact, not a product.** ~114M parameters, trained on 500M tokens
> (well under Chinchilla-optimal for this size). It writes fluent, on-topic English
> but its facts are unreliable. Its purpose is a controlled scientific comparison.

## This model

| | |
|---|---|
| Role | {f['role']} |
| Recurrence (`use_loop`) | {loop_txt} |
| Clifford GDR (`use_gdr`) | {gdr_txt} |
| Parameters | {f['params']} |
| Checkpoint | `step-00007630.pt` (model + optimizer state + config dict) |
| Tokens seen | 500M (sequence length 1024) |

## What is HAGI / GDR?

HAGI tests one hypothesis: does decomposing a recurrent transformer's hidden state
into **Clifford-algebra grades** - scalar / vector / bivector / trivector, each with
its own update rate, plus a geometric-product cross-grade interaction - improve
reasoning-per-parameter in small models?

A standard recurrent-depth transformer iterates over a flat hidden vector and gains
plateau after a few iterations because every dimension converges at the same rate.
GDR splits the state into grades that evolve at different rates, so each reasoning
iteration has distinct dynamics.

## The ablation (where this model fits)

| Model | Recurrence | Clifford GDR | Params |
|-------|-----------|--------------|--------|
{_ablation_table(m)}

**Decisive comparison: B vs D** - identical parameters and compute pattern; the only
difference is grade decomposition. Secondary: **C vs D** (integrated vs bolted-on
Clifford). A is the floor. Read with the gates in `docs/ABLATION.md`.

## Architecture (shared by all four)

| Component | Value |
|-----------|-------|
| Hidden size | 768 |
| Layers | 12 (4 perception / 4 reasoning / 4 expression) |
| Attention | Grouped-Query Attention (12 query heads, 4 KV heads) |
| MLP | SwiGLU (768 -> 2048 -> 768) |
| Positional | RoPE (theta=10000) |
| Norm | RMSNorm (pre-norm) |
| Embeddings | weight-tied input/output |
| Vocabulary | 49,152 (SmolLM2 tokenizer) |
| Sequence length | 1024 |
| Precision | bf16 |

Models **C** and **D** add the GDR grade-update MLPs + geometric product (+~1.3M
params -> 114.6M). Models **B** and **D** loop the reasoning core **3x** per forward.

## Training (shared)

| | |
|---|---|
| Data | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample-10BT` |
| Tokens | 500M (~7,629 steps) |
| Optimizer | AdamW (lr 3e-4, wd 0.1, cosine decay, 400-step warmup) |
| Effective batch | 65,536 tokens/step (batch 16 x grad-accum 4 x seq 1024) |
| Hardware | Google Colab A100-40GB (bf16 + FlashAttention + `torch.compile`) |

## Load and run (free, CPU)

```bash
git clone -b experimental https://github.com/ShmidtS/HAGI.git && cd HAGI
pip install -r requirements.txt
python scripts/generate.py --hf-repo {user}/hagi-ablation-{m} \\
    --prompt "The sun is a star that" --device cpu
```

Inference fits in <1GB - no GPU needed. Or load the checkpoint directly:

```python
import torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download
from prototype.training.loop import load_checkpoint
from prototype.data.tokenizer import load_tokenizer

ckpt = hf_hub_download("{user}/hagi-ablation-{m}", "step-00007630.pt")
model, step = load_checkpoint(ckpt, device="cpu"); model.eval()
tok = load_tokenizer("HuggingFaceTB/SmolLM2-135M")

x = torch.tensor([tok.encode("The sun is a star that")])
for _ in range(60):
    with torch.no_grad():
        logits = model(x)[0, -1]
    x = torch.cat([x, torch.multinomial(F.softmax(logits / 0.8, -1), 1).view(1, 1)], dim=1)
print(tok.decode(x[0].tolist()))
```

## Sibling models

- **A** - dense baseline (control): `{user}/hagi-ablation-a`
- **B** - recurrence only: `{user}/hagi-ablation-b`
- **C** - Clifford, bolted-on: `{user}/hagi-ablation-c`
- **D** - full GDR: `{user}/hagi-ablation-d`

## Reading the result

Score the four on a shared fixed batch set with `scripts/eval_loss.py`; lower loss
is better. **D below B** => grade decomposition helps (scaling is then justified).
**D == B** => neutral. **D above B** => harmful. See `docs/ABLATION.md` for the gates.

## Links

- Code: https://github.com/ShmidtS/HAGI (branch `experimental`)
- Spec: `docs/ARCHITECTURE.md`, `docs/MILESTONES.md`, `docs/ABLATION.md`
- License: Apache 2.0
"""


def main():
    ap = argparse.ArgumentParser(description="Push a README.md model card to each ablation repo.")
    ap.add_argument("--user", required=True, help="HF username (repos: <user>/hagi-ablation-{a,b,c,d})")
    ap.add_argument("--models", default="a,b,c,d", help="comma-separated subset to push")
    ap.add_argument("--dry-run", action="store_true", help="print the first card instead of pushing")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.dry_run:
        print(card(models[0], args.user))
        return

    from huggingface_hub import HfApi

    api = HfApi()
    for m in models:
        repo = f"{args.user}/hagi-ablation-{m}"
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=io.BytesIO(card(m, args.user).encode("utf-8")),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="model",
        )
        print(f"pushed README.md -> {repo}")


if __name__ == "__main__":
    main()
