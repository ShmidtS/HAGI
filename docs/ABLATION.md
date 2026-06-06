# GDR Ablation — Run Guide

Four same-size (~115M) models, identical data / schedule / tokenizer / budget,
differing only in `use_loop` and `use_gdr`. This is the core HAGI experiment.

| Config | Loop | GDR | Params | Isolates |
|--------|------|-----|--------|----------|
| `ablation_a` | ❌ | ❌ | 113.3M | dense baseline (control) |
| `ablation_b` | ✅ ×3 | ❌ | 113.3M | recurrence alone |
| `ablation_c` | ❌ | ✅ | 114.6M | Clifford bolted on |
| `ablation_d` | ✅ ×3 | ✅ | 114.6M | full GDR (loop + grades) |

**Decisive comparison: B vs D** — same params and compute pattern, the only
difference is grade decomposition. Secondary: **C vs D** (integrated vs bolted-on
Clifford). A is the floor.

## Scale (deliberately small)

seq 1024, 500M tokens, effective batch 65,536. The comparison is *relative*, so
absolute scale matters less than keeping all four identical. Reuses the same Drive
shards as Stage 0 (the loader windows them to 1024) — **no re-tokenize**.

## Cost & time (A100, ~12 units/hr; estimates)

| Model | ~time | ~units |
|-------|-------|--------|
| A (no loop) | ~0.9h | ~11 |
| B (loop ×3) | ~1.6h | ~19 |
| C (no loop) | ~1.0h | ~12 |
| D (loop ×3) | ~1.6h | ~19 |
| **all four** | **~5h** | **~61** |

Looped models (B, D) are ~1.7× the compute — the ablation is *more* expensive than
the baseline, not less. **One GPU runs one model at a time** (no on-GPU
parallelism); overlap two by running one on Colab and one on Kaggle.

**Budget note:** all four at 500M ≈ ~61 units. If you have ~40:
- drop `train_tokens` to `300_000_000` (~37 units, all four — noisier signal), or
- run **B + D first** (the key pair, ~38 units) and add A/C later, or
- top up ~$10 (→ ~100 units) for a cleaner 500M (or 1B) run.

## Run order

Each model is a separate run with its own checkpoint dir + HF repo:

```bash
DATA=/content/drive/MyDrive/hagi-data   # the seq-1024-compatible Stage 0 shards

python -m prototype.training.train --config configs/ablation_a.yaml --data $DATA \
    --device cuda --hf-repo NAME0x0/hagi-ablation-a --resume auto --steps 8000
python -m prototype.training.train --config configs/ablation_b.yaml --data $DATA \
    --device cuda --hf-repo NAME0x0/hagi-ablation-b --resume auto --steps 8000
python -m prototype.training.train --config configs/ablation_c.yaml --data $DATA \
    --device cuda --hf-repo NAME0x0/hagi-ablation-c --resume auto --steps 8000
python -m prototype.training.train --config configs/ablation_d.yaml --data $DATA \
    --device cuda --hf-repo NAME0x0/hagi-ablation-d --resume auto --steps 8000
```

(`--steps 8000` ≥ max_steps 7629, so each finishes its full 500M in one session;
re-run with `--resume auto` if a session dies.)

## Analysis (after all four train)

1. **Final train/val loss** per model → perplexity.
2. **Benchmarks** via the eval adapter (cheap, runs locally on your 4GB):
   ```bash
   python -m prototype.evaluation.evaluate --ckpt checkpoints/ablation_d/step-00007629.pt \
       --benchmarks arc_challenge,boolq,hellaswag,winogrande --device cuda
   ```
3. **Gate:** D beats both B and C on ≥2 reasoning benchmarks by ≥2% absolute, and
   D's perplexity is within ~3% of A.
4. **Read the gates** (see `docs/MILESTONES.md` Stage 2):
   - **D ≈ B** → grade decomposition is neutral. Publish B + recipe.
   - **D < B** → harmful; revisit momentum / grade partition / residual size.
   - **D > B** → the result. Grade decomposition adds value → scale (MoE, params,
     tokens) is now justified.

Comparison is cheap and local; only the four training runs need the GPU.
