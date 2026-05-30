# HAGI PyTorch Prototype

Primary development surface. The PyTorch prototype is the source of truth during
research (Stages 0-4). The Rust port (`../crates/`) happens only after Stage 2
validates the architecture.

## Layout

```
prototype/
├── model/
│   ├── clifford.py      # Cl(3,0,0) geometric product (verified, tested)
│   ├── transformer.py   # RMSNorm, RoPE, GQA, SwiGLU, TransformerBlock
│   ├── gdr.py           # Grade-Decomposed Recurrence (the novel mechanism)
│   └── hagi.py          # Full model — all 4 ablation variants via flags
├── data/
│   ├── tokenizer.py     # SmolLM2 (49K) wrapper
│   ├── tokenize.py      # datatrove -> .bin shards (FineWeb-Edu/code/math)
│   ├── dataset.py       # memmap get_batch (nanoGPT-style)
│   └── toy.py           # in-memory toy corpus for overfit test
├── training/
│   ├── config.py        # YAML -> typed config
│   ├── optim.py         # AdamW + Muon hybrid optimizer
│   ├── loop.py          # core training loop (bf16, accum, cosine LR, ckpt, eval)
│   └── train.py         # CLI driver
├── evaluation/
│   ├── lm_eval_wrapper.py  # HAGI adapter for lm-eval-harness
│   └── evaluate.py         # benchmark CLI + intelligence-density metrics
└── tests/               # Clifford + model + overfit tests
```

See [../docs/TRAINING.md](../docs/TRAINING.md) for the full training workflow.

## Ablation Models

One class, four configs (see `hagi.py`):

| Model | `use_loop` | `use_gdr` | Description |
|-------|-----------|-----------|-------------|
| A | False | False | Dense baseline (control) |
| B | True | False | Looped, flat recurrence |
| C | False | True | Clifford bolted on, no loop |
| D | True | True | Full GDR (the bet) |

## Quick Start

```bash
pip install -r ../requirements-dev.txt

# Run tests (Clifford math + model forward/backward)
python -m pytest prototype/tests/ -q

# Count parameters for a config
python scripts/param_count.py --config configs/gdr.yaml

# Train (once data pipeline is wired)
python -m prototype.training.train --config configs/gdr.yaml
```

## Status

- [x] Clifford `Cl(3,0,0)` geometric product — implemented, 8 tests pass
- [x] Transformer blocks (GQA, SwiGLU, RoPE, RMSNorm)
- [x] Grade-Decomposed Recurrence module
- [x] Full model — all 4 variants instantiate, forward + backward verified
- [x] Optimizer — AdamW + Muon hybrid (orthogonalized updates)
- [x] Training loop — nanoGPT-adapted, overfit test passes (AdamW + Muon)
- [x] Data loader — memmap shards + toy corpus
- [x] lm-eval-harness adapter — loglikelihood + generate_until
- [ ] Run datatrove tokenization on FineWeb-Edu (needs data download)
- [ ] Stage 0 baseline training run

## Optimizer Note

Baselines use AdamW (clean control). Muon (orthogonalized updates, powers the
nanoGPT speedrun) is a separate ablation — set `optimizer: muon` in the config.
Do not change optimizer and architecture in the same comparison.

## Tests Mirror Lean4

`tests/test_clifford.py` checks the same invariants proven in
`../formalization/HAGI/HDIM.lean` (grade lookup, scalar identity, anticommutation,
pseudoscalar). This is the Stage 6 verification bridge, started early.
