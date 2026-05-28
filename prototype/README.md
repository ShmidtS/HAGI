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
├── data/                # Data pipeline (TODO Stage 0)
├── training/
│   ├── config.py        # YAML -> typed config
│   └── train.py         # Training driver
├── evaluation/
│   └── evaluate.py      # Benchmark harness + intelligence-density metrics
└── tests/               # Clifford + model smoke tests
```

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
- [ ] Data pipeline (tokenizer, sharding, PrefixLM packing)
- [ ] Training loop (skeleton exists, needs data)
- [ ] Benchmark runners (metric helpers exist, need lm-eval integration)

## Tests Mirror Lean4

`tests/test_clifford.py` checks the same invariants proven in
`../formalization/HAGI/HDIM.lean` (grade lookup, scalar identity, anticommutation,
pseudoscalar). This is the Stage 6 verification bridge, started early.
