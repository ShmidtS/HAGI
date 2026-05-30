<p align="center">
  <h1 align="center">HAGI</h1>
  <p align="center"><strong>Hypercomplex Artificial General Intelligence</strong></p>
  <p align="center">
    A research architecture exploring grade-decomposed Clifford recurrence for intelligence-dense small language models.
  </p>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
  <a href="CODE_OF_CONDUCT.md"><img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"></a>
</p>

---

## What Is HAGI?

HAGI is a research project investigating whether **Clifford algebra grade structure** can improve iterative reasoning in small language models. The core hypothesis:

> Standard recurrent-depth transformers iterate over flat vector representations — every dimension updates at the same rate, leading to diminishing returns after a few iterations. HAGI decomposes the hidden state into **Clifford grades** (scalars, vectors, bivectors, trivectors) where each grade carries semantically different information and evolves at a different rate during recurrence. The geometric product provides structured cross-grade interaction, giving each iteration of reasoning fundamentally different dynamics than the last.

This is **not** an attempt to build a frontier LLM. It is a controlled research experiment to answer: *Does geometric structure in the recurrence representation measurably improve reasoning in small models?*

## Core Architecture: Grade-Decomposed Recurrence (GDR)

The model follows a **Perception → Reasoning → Expression** pipeline:

```
Input Tokens
      │
      ▼
┌─────────────────────────────────────────────┐
│  PERCEPTION (Layers 1-4, unique params)     │
│  Standard transformer blocks.               │
│  Maps tokens → rich contextual embeddings.  │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  REASONING CORE (Layers 5-8, LOOPED 3-5×)  │
│                                             │
│  Each iteration:                            │
│   1. Decompose hidden → Clifford grades     │
│   2. Grade-specific update:                 │
│      • Scalars: slow momentum (α=0.9)       │
│      • Vectors: medium momentum (α=0.5)     │
│      • Bivectors: full update (reasoning)   │
│      • Trivectors: full update (structure)  │
│   3. Geometric product: cross-grade mixing  │
│   4. Recompose → hidden state               │
│   5. Standard transformer attention + MLP   │
│   6. Add iteration embedding                │
│                                             │
│  Parameters shared across all iterations.   │
│  ~115M params; effective depth ~20 layers.  │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  EXPRESSION (Layers 9-12, unique params)    │
│  Standard transformer blocks.               │
│  Refines representations → logits.          │
└─────────────────────────────────────────────┘
      │
      ▼
   LM Head → Logits
```

### Why Grade Decomposition?

| Approach | Representation | Grade Awareness | Problem |
|----------|---------------|-----------------|---------|
| Standard Transformer | Flat vector | None | Fixed depth, no iteration |
| Looped Transformer (Huginn) | Flat vector, iterated | None | Diminishing returns — all dims converge at same rate |
| HRM (H/L modules) | Two separate flat vectors | Architectural split | Parameter duplication |
| **HAGI (GDR)** | **Grade-structured multivector** | **Per-grade dynamics** | **Novel — under investigation** |

The geometric product of `Cl(3,0,0)` naturally mixes grades: `vector × vector → scalar + bivector`. This means entity-level reasoning automatically generates relational and confidence signals without requiring separate learned mechanisms.

## Research Status

> **Phase: Prototype built, untrained.** The PyTorch prototype — model, training stack, and evaluation adapter — exists and passes correctness tests. No model has been trained yet; the next step is the Stage 0 baseline run.

### What Exists

- PyTorch prototype: GDR model with Perception/Reasoning/Expression, all four ablation variants behind config flags (`prototype/model/`)
- Training stack: nanoGPT-adapted loop, AdamW+Muon optimizer, datatrove tokenization, memmap loader, lm-eval-harness adapter (`prototype/training/`, `prototype/data/`, `prototype/evaluation/`)
- Test suite: Clifford algebra, model shape, overfit, config smoke, checkpoint roundtrip (20 tests, CPU-only)
- Rust workspace scaffold (10 crates) with typed Clifford primitives (refactored for GDR at Stage 5)
- Lean4 formal verification of core invariants (~700 lines, aligned at Stage 6)
- Architecture documentation, research analysis, milestone-driven plan

### What's Next

- Stage 0: tokenize FineWeb-Edu, train the dense baseline (Model A)
- Stages 1-2: recurrent core, then the full GDR ablation (Models A/B/C/D)

## Key Design Decisions

1. **PyTorch first, Rust later.** Hypothesis validation requires fast iteration. The Rust implementation is for production, not prototyping.

2. **`Cl(3,0,0)` (8 blades).** Pragmatic choice: 64 multiplications per geometric product. Large enough for meaningful grade decomposition (4 grades), small enough to be negligible compute overhead.

3. **Residual channel.** 33% of hidden dimensions are unconstrained by grade structure. This safety valve lets the model bypass grade decomposition when it doesn't help.

4. **No MoE, no memory, no sparse attention in the prototype.** Each is a separate research variable. They would confound the core experiment.

5. **Formal verification for Clifford operations only.** Lean4 proofs catch algebraic bugs. Verifying the training loop adds no value until a model trains successfully.

## Experimental Plan

Four models, identical training, architecture-only differences:

| Model | Architecture | Purpose |
|-------|-------------|---------|
| **A** (Baseline) | 12-layer dense transformer | Control |
| **B** (Loop) | Layers 5-8 looped 3×, flat hidden state | Isolate recurrence benefit |
| **C** (HDIM) | 12-layer dense + Clifford projection layers | Isolate Clifford benefit |
| **D** (GDR) | Layers 5-8 looped 3× with grade-decomposed recurrence | **Full HAGI architecture** |

**Success criterion:** Model D outperforms both B and C on reasoning benchmarks (GSM8K, ARC-Challenge, BoolQ) while not degrading perplexity.

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Unique parameters | ~115M |
| Effective depth (3× loop) | 20 layers |
| Hidden size | 768 |
| Attention | GQA (12 query heads, 4 KV heads) |
| MLP | SwiGLU (768 → 2048 → 768) |
| Position encoding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| Context length | 4096 tokens |
| Vocabulary | 49,152 (SmolLM2 BPE) |
| Clifford algebra | `Cl(3,0,0)`, 8 blades |
| Grade allocation | 64 scalar + 192 vector + 192 bivector + 64 trivector + 256 residual = 768 |
| Training precision | bf16 |
| Deployment target | Q4_K_M (GGUF), ~65MB |

## Project Structure

```
HAGI/
├── prototype/              # PyTorch prototype (primary development)
│   ├── model/              # clifford, transformer, gdr, hagi
│   ├── data/               # tokenizer, datatrove tokenize, memmap dataset, toy
│   ├── training/           # optim (AdamW+Muon), loop, train CLI, config
│   ├── evaluation/         # lm-eval adapter + metrics
│   └── tests/              # clifford, model, overfit
├── crates/                 # Rust implementation (Stage 5+)
│   ├── clifford-core/      # Clifford algebra primitives
│   ├── core-types/         # Shared type definitions
│   ├── tensor-runtime/     # Tensor substrate
│   ├── hrm-model/          # HRM backbone
│   ├── hdim-model/         # HDIM structural layer
│   └── ...
├── formalization/          # Lean4 formal verification
│   ├── HAGI/
│   │   ├── CoreTypes.lean
│   │   ├── HDIM.lean
│   │   ├── HRM.lean
│   │   └── ...
│   └── lakefile.lean
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # Detailed architecture specification
│   ├── TRAINING.md         # Training stack + workflow
│   ├── MILESTONES.md       # Milestone definitions and tracking
│   ├── RESEARCH.md         # Research background and references
│   └── ...
├── benchmarks/             # Benchmark scripts and results
├── configs/                # Model and training configurations
└── scripts/                # Utility scripts
```

## Milestones

| Stage | Name | Goal | Status |
|-------|------|------|--------|
| **0** | Dense Baseline | Working 115M transformer, trained, benchmarked | Not started |
| **1** | Recurrent Core | Looped layers 5-8 × 3, flat recurrence | Not started |
| **2** | Grade-Decomposed Recurrence | Clifford grade decomposition in reasoning loop | Not started |
| **3** | Context Efficiency | MLA KV compression, extended context | Not started |
| **4** | Quantization Path | Q4/Q2 deployment, quality retention | Not started |
| **5** | Rust/CUDA Port | Production implementation of validated architecture | Not started |
| **6** | Formal Verification | Lean4 ↔ Rust property test alignment | Not started |

See [docs/MILESTONES.md](docs/MILESTONES.md) for detailed definitions, acceptance criteria, and stop/pivot conditions.

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- (Optional) Rust 1.78+ for crate development
- (Optional) Lean4 / Lake for formal verification

### Setup

```bash
# Clone
git clone https://github.com/ShmidtS/HAGI.git
cd HAGI

# Python environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# (Optional) Rust workspace
cargo check --workspace

# (Optional) Lean4 formalization
cd formalization && lake build
```

### Training

See [docs/TRAINING.md](docs/TRAINING.md) for the full workflow. Quick version:

```bash
# 0. Overfit sanity check (no data needed — proves the loop is correct)
python -m pytest prototype/tests/test_overfit.py -q

# 1. Tokenize a corpus into shards (datatrove)
python -m prototype.data.tokenize --dataset HuggingFaceFW/fineweb-edu \
    --subset sample-10BT --output data/fineweb-edu --tokenizer HuggingFaceTB/SmolLM2-135M

# 2. Train (Stage 0 baseline, then Stage 2 GDR)
python -m prototype.training.train --config configs/baseline.yaml --data data/fineweb-edu
python -m prototype.training.train --config configs/gdr.yaml --data data/fineweb-edu
```

### Evaluation

```bash
python -m prototype.evaluation.evaluate \
    --ckpt checkpoints/gdr/step-00050000.pt \
    --benchmarks gsm8k,arc_challenge,boolq
```

## Research Context

HAGI builds on ideas from multiple research directions. See [docs/RESEARCH.md](docs/RESEARCH.md) for the full literature review. Key influences:

- **Recurrent-depth transformers** — Huginn, LoopLM, Universal Transformer
- **Clifford algebra in ML** — GATr, CliffordNet, CGENNs
- **Efficient attention** — DeepSeek MLA, Native Sparse Attention
- **Small model training** — Phi-4, SmolLM-3, knowledge distillation
- **Adaptive computation** — PonderNet, ACT

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas where help is needed:
- PyTorch prototype implementation
- Benchmark infrastructure
- Clifford algebra kernel optimization
- Lean4 formalization expansion
- Training data curation

## Citation

```bibtex
@software{hagi2025,
  title  = {HAGI: Hypercomplex Artificial General Intelligence},
  author = {HAGI Contributors},
  url    = {https://github.com/ShmidtS/HAGI},
  year   = {2025}
}
```

## License

[Apache License 2.0](LICENSE)

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
