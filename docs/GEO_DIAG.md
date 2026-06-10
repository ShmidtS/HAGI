# Geo-Ablation Diagnostic — Results Record (2026-06-09/10)

Follow-up to the seed-stability gate in `docs/ABLATION.md`. Question: GDR (model D)
lost to recurrence-only (model B) on held-out loss, 0/5 seeds — is the
**geometric-product cross-grade term** the culprit, or the **grade decomposition
itself**?

## Setup (everything matched)

| | |
|---|---|
| Platform | Kaggle, 1× Tesla T4 16GB, fp16 (GradScaler), `torch.compile` |
| Variants | B (recurrence only) / D (full GDR) / **D_nogeo** (`grades.geo_interaction: false`, same 114.63M params as D) |
| Tokens per run | 120,000,000 (1,832 steps × 65,536 tok eff. batch = batch 4 × accum 16 × seq 1024) |
| Seed | 1 (single seed; prior gate was 5/5 consistent) |
| Data | `NAME0x0/hagi-fineweb-edu-smollm2` (HF dataset) — train shards 00000–00005 |
| Held-out val | `shard_00006.bin` (same val shard as the seed-stability gate) |
| Eval | `scripts/eval_loss.py`, 50 × 16 × 1024 = 819,200 tokens, eval seed 1234 |
| Notebook | `notebooks/hagi_kaggle_geo_diag.ipynb` (reproduces this end-to-end) |
| Throughput | ~11.3–11.6k tok/s steady-state per run (~2.9 h/run) |

## Results (held-out loss, lower is better)

| Variant | Loss | PPL | Δ vs B |
|---------|------|-----|--------|
| **B** — recurrence only | **4.2581** | 70.68 | — |
| D — full GDR | 4.2792 | 72.18 | **+0.0210** |
| D_nogeo — grades, no geometric product | 4.2757 | 71.93 | **+0.0176** |

Raw eval JSON:

```json
{"ablation_b": {"step": 1832, "loss": 4.25814, "ppl": 70.6784},
 "ablation_d": {"step": 1832, "loss": 4.279187, "ppl": 72.1817},
 "ablation_d_nogeo": {"step": 1832, "loss": 4.275744, "ppl": 71.9337}}
```

`GEO_DIAG seeds=[1] mean_D-B=+0.0210 mean_Dnogeo-B=+0.0176`

## Read

- D−B = **+0.0210** reproduces the negative on this platform and matches the prior
  5-seed mean (+0.0175). Consistent across hardware, precision, and seeds.
- Switching the geometric product OFF recovered only **~16%** of the gap
  (0.0210 → 0.0176). **The geometric product is not the culprit — the grade
  decomposition machinery itself hurts at this scale.**
- D and D_nogeo carry +1.3M parameters over B and lose on **both** train and
  held-out: a harmful inductive bias, not mere overfitting.

**Conclusion: GDR-as-built (grades-as-hidden-state-partition + auxiliary geometric
term) is falsified for language pretraining at ~114M / 120M tokens.** Combined with
the 0/5 seed gate, the project pivots to capability (fine-tuning a proven open base);
model B — a plain 113.3M recurrent transformer — is the retained baseline.

## Artifact note

The T4 checkpoints (`/kaggle/working/ckpt/...`) were **not retained** — the Kaggle
session expired (>40 min idle) before they could be uploaded, and `/kaggle/working`
does not survive session shutdown. The recorded losses above are the scientific
result; the published model weights remain the original A100 ablation checkpoints
on HF (`NAME0x0/hagi-ablation-{a,b,c,d}`). Future long runs should pass
`--hf-repo` so `train.py` mirrors checkpoints to HF as they are written.

This verdict ships in every ablation model card via the `GEO_DIAG` constant in
`scripts/push_model_cards.py`.
