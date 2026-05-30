# HAGI Training Stack

HAGI does not hand-roll training infrastructure. It composes battle-tested
components and wraps them around the custom GDR model. This document is the
operational guide; for the architecture see [ARCHITECTURE.md](ARCHITECTURE.md),
for the staged plan see [MILESTONES.md](MILESTONES.md).

## Read First: The Smol Training Playbook

Before training, read HuggingFace's **[Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)**
(SmolLM2/3 team, Oct 2025). It is the practical recipe for ~100M-3B models:
data mix, hyperparameters, ablation methodology, and the debugging reality.
Lock configs from it, not from guesses.

## The Stack

| Layer | Component | Why |
|-------|-----------|-----|
| Recipe | Smol Training Playbook | Configs/hyperparams/data-mix for this exact scale |
| Tokenizer | SmolLM2 (~49K vocab) | Trained on edu+code+math; right vocab size for 100M (embedding ~33%, not ~50%) |
| Data processing | `datatrove` | Tokenize + shard FineWeb-Edu / code / math into .bin streams |
| Data loading | `prototype/data/dataset.py` | nanoGPT-style memmap `get_batch` |
| Training loop | `prototype/training/loop.py` | nanoGPT-adapted: bf16 AMP, grad accum, cosine+warmup, clip, ckpt, eval |
| Optimizer | AdamW (baseline) / Muon (ablation) | Muon orthogonalizes updates; faster convergence on FineWeb small-model |
| Evaluation | `lm-eval-harness` via HAGI adapter | Standard GSM8K/ARC/BoolQ/... |

Design choice: we use these frameworks' **infrastructure**, not their models. The
GDR architecture is custom and lives in `prototype/model/`. The loop wraps it
untouched — `get_batch()` is the only contract.

## Workflow

### 1. Tokenize a corpus → shards

```bash
python -m prototype.data.tokenize \
    --dataset HuggingFaceFW/fineweb-edu \
    --subset sample-10BT \
    --output data/fineweb-edu \
    --tokenizer HuggingFaceTB/SmolLM2-135M
```

Repeat per source (code, math) and combine, weighting via the playbook's mix.

### 2. Overfit sanity check (always do this first)

```bash
python -m pytest prototype/tests/test_overfit.py -q
```

A tiny model must drive loss → ~0 on a fixed toy corpus. If this fails, the loop
is broken — fix before touching real data. Covers AdamW and Muon.

### 3. Train

```bash
# Stage 0 — dense baseline (Model A, the control)
python -m prototype.training.train --config configs/baseline.yaml --data data/fineweb-edu

# Stage 2 — full GDR (Model D)
python -m prototype.training.train --config configs/gdr.yaml --data data/fineweb-edu
```

### 4. Evaluate

```bash
python -m prototype.evaluation.evaluate \
    --ckpt checkpoints/gdr/step-00050000.pt \
    --benchmarks gsm8k,arc_challenge,boolq,hellaswag,winogrande
```

## Checkpoints

`save_checkpoint(model, optimizer, step, ckpt_dir)` writes `step-<N>.pt` into
`checkpoints/<config-name>/`. The model config is stored as a plain dict (not a
pickled dataclass), so checkpoints load under torch's default `weights_only=True`
— no arbitrary code execution on load.

Load for resume or evaluation:

```python
from prototype.training.loop import load_checkpoint
model, step = load_checkpoint("checkpoints/baseline/step-00050000.pt", device="cuda")
```

The eval adapter (`prototype/evaluation/lm_eval_wrapper.py`) loads via this path,
so `--ckpt` accepts any saved checkpoint directly.

## Optimizer: AdamW vs Muon

- **Baseline runs use AdamW.** It is the clean control.
- **Muon is a separate ablation variable.** Do not change optimizer and
  architecture in the same comparison — that confounds the result.
- Muon applies to 2D weight matrices (attention/MLP/GDR-MLP weights). Embeddings,
  LM head, norms, gates, and iteration embeddings stay on AdamW (the hybrid is
  built automatically by `build_optimizer`).
- Muon pairs with **bf16** (default). fp16 + Muon is unsupported by the loss
  scaler path — use bf16 or fp32.

Set `optimizer: muon` in the config to enable.

## The Ablation

Four models, identical data/schedule/tokenizer, architecture-only differences:

| Model | `use_loop` | `use_gdr` | Tests |
|-------|-----------|-----------|-------|
| A | false | false | baseline |
| B | true | false | recurrence only |
| C | false | true | Clifford bolted-on |
| D | true | true | full GDR |

Decisive comparison: **B vs D**. Same params, same compute pattern; the only
difference is grade decomposition. See [MILESTONES.md](MILESTONES.md) Stage 2 for
gates and pivot conditions.

## Hardware Notes

- ~100M model trains on a single 24GB GPU (4090/A5000/L4) in bf16.
- The full Stage 0 config (`batch_size=16`, `max_seq_len=4096`) needs ~24GB. The
  49K-vocab logits at length 4096 alone are multiple GB; a small laptop GPU (≤8GB)
  cannot run it. On such hardware, use only the overfit test, or a reduced **dev
  profile** (short `max_seq_len`, `batch_size` 1-2, higher `grad_accum_steps`) to
  validate the pipeline end-to-end — not the science. Real runs go on a rented/
  cloud 24GB+ GPU.
- Adjust `batch_size` × `grad_accum_steps` to fit memory while holding the
  effective batch (tokens/step) constant across all four ablation models.
- CPU is fine for the overfit test only.
