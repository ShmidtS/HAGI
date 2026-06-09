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

# ruff: noqa: E501 — card text + sample generations are intentionally long string literals
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

# Measured results - first pass: one run per model (training seed 42), scored on a
# shared fixed batch set (identical batches for all four). Update after seed runs.
RESULTS = {
    "a": dict(loss=3.4880, ppl=32.72, sample="The sun is a star that rises in the sky, then rises in the sky, eventually the sun. The two stars, the sun and the moon, are equally the same but the earth is smaller in size and has more diameter"),
    "b": dict(loss=3.4852, ppl=32.63, sample="The sun is a star that rises in the sky, then rises again in the sky as the sun approaches the Sun. The sun itself does not fall in the sky because of its position above the horizon. Some of the planets"),
    "c": dict(loss=3.4771, ppl=32.37, sample="The sun is a star that is the main star in the picture. It is an extremely hot star that can be seen through the sun. This is also the brightest star in the picture. The sun is another star that can"),
    "d": dict(loss=3.4702, ppl=32.14, sample="The sun is a star that has a half-life of 8.6 hours. The Sun is 11.86 million miles tall and is the tallest red dwarf in the world. It is a binary star"),
}
TRAIN_SEED = 42
EVAL_TOKENS = "819,200"

# Seed-stability: B vs D across 5 seeds at 120M tokens on a held-out shard.
# Result: D beat B in 0/5 — grade decomposition consistently HURTS held-out loss.
SEED_STABILITY = dict(seeds=[1, 2, 3, 4, 5], d_below_b=0, mean_d_minus_b=0.0175, val="shard_00006.bin")


def _results_table(this: str) -> str:
    rows = []
    for k in ORDER:
        r = RESULTS[k]
        label = f"**{k.upper()} (this)**" if k == this else k.upper()
        rows.append(f"| {label} | {MODELS[k]['params']} | {r['loss']:.4f} | {r['ppl']:.2f} |")
    return "\n".join(rows)


def _seed_block() -> str:
    if not SEED_STABILITY:
        return ("**Seed stability:** the B-vs-D check across multiple seeds on a held-out "
                "shard is in progress; this card will be updated with the per-seed verdict.")
    s = SEED_STABILITY
    n = len(s["seeds"])
    return (f"**Seed stability:** across {n} seeds on held-out validation ({s.get('val', 'a held-out shard')}), "
            f"**D beat B in {s['d_below_b']}/{n}** runs (mean loss D-B = {s['mean_d_minus_b']:+.4f}).")


def _ablation_table(this: str) -> str:
    rows = []
    for k in ORDER:
        loop = "loop x3" if MODELS[k]["loop"] else "-"
        gdr = "grades" if MODELS[k]["gdr"] else "-"
        label = f"**{k.upper()} (this model)**" if k == this else k.upper()
        rows.append(f"| {label} | {loop} | {gdr} | {MODELS[k]['params']} |")
    return "\n".join(rows)


def _family_block(user: str, this: str) -> str:
    """Clickable nav across the whole model family (Stage 0 + the four ablations)."""
    base = f"https://huggingface.co/{user}"
    links = [("Stage 0", "pretraining baseline (separate track, not part of this ablation)", f"{base}/hagi-stage0")]
    for k in ORDER:
        links.append((k.upper(), MODELS[k]["role"], f"{base}/hagi-ablation-{k}"))
    rows = []
    for label, role, url in links:
        mark = " - **you are here**" if label.lower() == this else ""
        rows.append(f"| [{label}]({url}){mark} | {role} |")
    return "\n".join(rows)


def _geo_block(geo: dict | None) -> str:
    """Optional geometry-diagnostic section (filled in after the Kaggle T4 run)."""
    if not geo:
        return ""
    dnb, dngb = geo["dnb"], geo["dngb"]
    seeds = geo.get("seeds", [])
    n = len(seeds)
    if dnb > 0 and dngb <= 0.25 * dnb:
        read = ("Removing the geometric product recovers most of the gap, so the **geometric-product "
                "cross-grade term was the main culprit** - not the grade split itself. Next bet: a "
                "paper-faithful geometric design rather than grades-as-auxiliary.")
    elif dngb > 0:
        read = ("Removing the geometric product does **not** recover the gap - the **grade machinery "
                "itself hurts** at this scale, independent of the geometric product. GDR-as-built is "
                "falsified here; the path forward is a paper-faithful rebuild or a pivot to capability.")
    else:
        read = ("Grades without the geometric product **beat** B - an unexpected positive that warrants "
                "a focused follow-up.")
    return f"""

## Geometry diagnostic (follow-up)

To locate *why* full GDR (D) lost to recurrence-only (B), a third arm - **`D_nogeo`**,
GDR with the geometric-product cross-grade term switched **off** (same {MODELS['d']['params']}
parameters as D) - was trained head-to-head with B and D on the same held-out shard
(Kaggle T4, fp16, {n} seed{'s' if n != 1 else ''}: {seeds}).

| Comparison | Mean held-out loss delta (lower = better) |
|-----------|-------------------------------------------|
| D - B (full GDR vs recurrence-only) | {dnb:+.4f} |
| D_nogeo - B (grades, no geometric product) | {dngb:+.4f} |

{read}
"""


def card(m: str, user: str, geo: dict | None = None) -> str:
    f = MODELS[m]
    loop_txt = "loop x3" if f["loop"] else "none"
    gdr_txt = "grades (scalar/vector/bivector/trivector + geometric product)" if f["gdr"] else "none"
    r = RESULTS[m]
    db = RESULTS["d"]["loss"] - RESULTS["b"]["loss"]
    dc = RESULTS["d"]["loss"] - RESULTS["c"]["loss"]
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

**TL;DR** - {f['role']}. One arm of a **four-model controlled ablation** testing whether
Clifford **grade decomposition** improves reasoning-per-parameter in a small language
model. **Headline result: it does not** - on held-out validation the plain recurrent
baseline (**B**) beats full GDR (**D**) on every seed. This card documents this arm and
links the rest of the family so you can traverse the whole experiment.

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
| Eval loss / perplexity | **{r['loss']:.4f} / {r['ppl']:.2f}** (shared {EVAL_TOKENS}-token set, seed {TRAIN_SEED}) |

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

## Model family (click to traverse)

All models below share data, tokenizer, schedule, and token budget. Stage 0 is the
separate pretraining-baseline track; A/B/C/D are this controlled ablation.

| Model | Role |
|-------|------|
{_family_block(user, m)}

## Results

All four scored on the **same** {EVAL_TOKENS}-token fixed batch set (identical
batches; lower loss is better). One run per model, training seed {TRAIN_SEED}.

| Model | Params | Loss | Perplexity |
|-------|--------|------|------------|
{_results_table(m)}

- **B vs D: loss D-B = {db:+.4f}** (negative = grade decomposition helps).
- **C vs D: loss D-C = {dc:+.4f}** (negative = integrated GDR beats bolted-on Clifford).
- Ordering **A < B < C < D** matches the hypothesis: the grade decomposition carries
  the signal; recurrence helps mainly in its presence (loop-alone B barely moves A).

**These are train-set numbers (one run per model).** They did NOT hold up: on
**held-out** validation across 5 seeds (note below), the ranking **reverses** - B
beats D every time (mean D-B = +0.0175). The small train-set D-advantage does not
generalize, consistent with mild overfitting by D's extra parameters. **Conclusion:
grade decomposition does not improve held-out loss at this scale.** A negative
result - see `docs/ABLATION.md` for the gates.

**This model's sample** (prompt *"The sun is a star that"*, temperature 0.8, seed {TRAIN_SEED}):

> {r['sample']}

{_seed_block()}
{_geo_block(geo)}
## Links

- Code: https://github.com/ShmidtS/HAGI (branch `experimental`)
- Spec: `docs/ARCHITECTURE.md`, `docs/MILESTONES.md`, `docs/ABLATION.md`
- License: Apache 2.0
"""


def main():
    ap = argparse.ArgumentParser(description="Push a README.md model card to each ablation repo.")
    ap.add_argument("--user", required=True, help="HF username (repos: <user>/hagi-ablation-{a,b,c,d})")
    ap.add_argument("--models", default="a,b,c,d", help="comma-separated subset to push")
    ap.add_argument("--geo-json", default=None,
                    help="optional JSON file with the geo-diagnostic verdict "
                         '(e.g. {"seeds":[1,2],"dnb":0.018,"dngb":0.005}); adds a Geometry section')
    ap.add_argument("--dry-run", action="store_true", help="print the first card instead of pushing")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    geo = None
    if args.geo_json:
        import json
        with open(args.geo_json, encoding="utf-8") as fh:
            geo = json.load(fh)
        print(f"geo-diagnostic loaded: D-B={geo['dnb']:+.4f}  D_nogeo-B={geo['dngb']:+.4f}")

    if args.dry_run:
        print(card(models[0], args.user, geo))
        return

    from huggingface_hub import HfApi

    api = HfApi()
    for m in models:
        repo = f"{args.user}/hagi-ablation-{m}"
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=io.BytesIO(card(m, args.user, geo).encode("utf-8")),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="model",
        )
        print(f"pushed README.md -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
