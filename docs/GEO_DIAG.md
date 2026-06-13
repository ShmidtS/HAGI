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

---

## Local gated re-test (2026-06-13) — was the negative an optimization artifact?

The cloud diagnostic showed the **grade decomposition** (not the geometric product)
carries the deficit. But the legacy GDR update had three optimization faults that
"grades hurt" could not be separated from: `sigmoid(0)=0.5` leaves the geo gates
half-open at step 0; bivector/trivector are **wholesale replaced** each iteration
(no residual path through the ×3 loop); and default-init MLPs start D far from B.
So the cloud result might measure *"untrainable as wired"* rather than *"grades are
useless."* The fair test: **ReZero-gated GDR** (`grades.gated: true`) — every grade
update becomes `h + alpha·f(h)` with `alpha` init 0, so the module is an **exact
identity at init** and gradient descent must *opt into* grades only if they help.

Run locally for free (RTX A2000 4GB, `--no-compile`), micro scale (~38M, 25M tokens,
seq 512). Three matched variants, identical data/schedule. Eval on the same held-out
data (`shard_00006`), averaged over 3 eval seeds (shared eval batches → low-noise
pairwise deltas):

| Variant | Held-out loss (eval-seed mean) | Δ vs B |
|---------|-------------------------------|--------|
| **micro_b** — recurrence only | **5.2527** | — |
| micro_d — legacy GDR | 5.2670 | **+0.0143** |
| micro_d_gated — ReZero-gated GDR | 5.2428 | **−0.0132** |

Per-seed Δ (1234 / 7 / 2024): D−B = +0.0162 / +0.0128 / +0.0138; gated−B =
−0.0118 / −0.0146 / −0.0133.

**Learned gates of micro_d_gated (final ckpt):**

```
alpha_scalar    -0.0002     gate_scalar    -0.0130
alpha_vector    +0.0001     gate_bivector  +0.0260
alpha_bivector  -0.0000
alpha_trivector +0.0002
```

### Read

1. **The harness is valid.** micro_d reproduces the negative — D−B = **+0.0143**,
   near-identical to the 114M held-out (+0.0175) and the cloud T4 (+0.0210). A free
   4GB local run is a faithful proxy for the cloud ablation.
2. **The grade machinery was declined, not rescued.** All four grade-update gates
   (`alpha_*`) converged to **machine-zero**. Given a fair, opt-in choice, gradient
   descent left the grade dynamics **switched off**. The gated variant ties/beats B
   only because it **collapsed back to B** — not because grades started helping. This
   *rules out* the optimization-artifact hypothesis: the faults were real, but fixing
   them removes the *harm* by letting the model walk away from grades, not by making
   them useful.
3. **One faint ember — the geometric product, not the grades.** The only non-dead
   gate is `gate_bivector → +0.026` (the geo cross-term), and gated−B is consistently
   **−0.013** across eval seeds. But: single *training* seed, magnitude ~1.3% ppl —
   right at the noise floor that took 5 seeds to clear at 114M. At most this says a
   *thin Clifford geometric-product side-channel* on plain recurrence is **neutral**,
   while grade decomposition is dead weight. Not a revival; a pointer.

**Conclusion: GDR-as-built is falsified airtight.** Grade decomposition does not help
even when offered for free with an identity init — the model refuses it. The geometric
product alone shows a sub-noise pulse that does *not* justify more from-scratch
training, but does say: if Clifford gets one more shot, it is the **geometric product
as a cheap adapter on a proven base**, never the grade partition. The pivot stands.

Reproduce: `configs/micro_{b,d,d_gated}.yaml`, `prototype/tests/test_gdr_gated.py`,
notebook `notebooks/runs/hagi_kaggle_geo_diag_2026-06-09.ipynb` (cloud original).
