<p align="center">
  <h1 align="center">HAGI</h1>
  <p align="center"><strong>Hypercomplex Artificial General Intelligence</strong></p>
  <p align="center">
    A research architecture exploring grade-decomposed Clifford recurrence, hierarchical reasoning, and geometric invariant transfer for intelligence-dense small language models.
  </p>
</p>

---

## What Is HAGI?

HAGI is a research project investigating whether **Clifford algebra grade structure** combined with **hierarchical recurrence** and **domain-invariant transfer** can improve iterative reasoning in small language models. The core hypothesis:

> Standard recurrent-depth transformers iterate over flat vector representations — every dimension updates at the same rate, leading to diminishing returns after a few iterations. HAGI decomposes the hidden state into **Clifford grades** (scalars, vectors, bivectors, trivectors) where each grade carries semantically different information and evolves at a different rate during recurrence. The geometric product provides structured cross-grade interaction, giving each iteration of reasoning fundamentally different dynamics than the last.

This is **not** an attempt to build a frontier LLM. It is a controlled research experiment to answer: *Does geometric structure in the recurrence representation measurably improve reasoning in small models?*

## Core Architecture

HAGI combines five novel mechanisms into a **Perception → Reasoning → Expression** pipeline:

| Mechanism | Abbreviation | Role |
|-----------|-------------|------|
| Grade-Decomposed Recurrence | GDR | Clifford grade-structured hidden state with per-grade update dynamics |
| Hierarchical Recurrent Model | HRM | Two-level reasoning loop (H-cycles × L-cycles) with strategic/tactical states |
| Hidden-state Decomposed Invariant Module | HDIM | Cross-domain invariant transfer via Clifford rotor sandwiches |
| Memory Sparse Attention | MSA | Slot-based sparse attention with external memory and HDIM routing |
| Mixture of Experts | MoE | SwiGLU expert routing with mixture-of-depths skip |

### Forward Pass Pipeline

```
Input Tokens + Position IDs
      |
      v
+---------------------------------------------+
|  PERCEPTION (2 layers, unique params)       |
|  Standard transformer blocks.               |
|  Maps tokens -> rich contextual embeddings. |
+---------------------------------------------+
      |
      v
+---------------------------------------------+
|  REASONING CORE (7 layers, HRM-controlled)  |
|                                             |
|  for h in 1..H_cycles:                      |
|    for l in 1..L_cycles:                    |
|      1. HDIM: project -> rotor -> invariant |
|      2. GDR: decompose -> grade update      |
|      3. Geometric product cross-grade mix   |
|      4. Recompose -> hidden state           |
|      5. Transformer block (GQA + MoE SwiGLU)|
|      6. MSA: slot routing + sparse attention|
|      7. H/L state transitions               |
|                                             |
|  ~74M params; 14 reasoning passes/step.     |
+---------------------------------------------+
      |
      v
+---------------------------------------------+
|  EXPRESSION (2 layers, unique params)       |
|  Standard transformer blocks.               |
|  Refines representations -> logits.         |
+---------------------------------------------+
      |
      v
   RMSNorm -> LM Head (576 -> 49K, weight-tied)
   or CAST: K-token block prediction via Cl(3,0,0)
```

### Why Grade Decomposition?

| Approach | Representation | Grade Awareness | Problem |
|----------|---------------|-----------------|---------|
| Standard Transformer | Flat vector | None | Fixed depth, no iteration |
| Looped Transformer (Huginn) | Flat vector, iterated | None | Diminishing returns — all dims converge at same rate |
| HRM (H/L modules) | Two separate flat vectors | Architectural split | Parameter duplication |
| **HAGI (GDR + HRM)** | **Grade-structured multivector** | **Per-grade dynamics** | **Novel — under investigation** |

The geometric product of `Cl(3,0,0)` naturally mixes grades: `vector x vector -> scalar + bivector`. This means entity-level reasoning automatically generates relational and confidence signals without requiring separate learned mechanisms.

## Additional Mechanisms

### CAST (Clifford Algebra Symbolic Reasoning Tokens)

Block-wise generation: each forward pass predicts K=8 tokens via multivector virtual states with geometric product coherence. Reduces sequential forward passes by 8x during generation.

### NARS (Non-Axiomatic Reasoning System)

Optional OpenNARS-style controllers that observe training signals (loss, gradient norms) and dynamically adjust HRM cycle counts, HDIM domain transfer, and MSA slot routing via truth revision and budget mechanisms. Disabled by default.

### Reasoning Cache (RC)

Iterative generate-summarize-cache decoding (arXiv:2602.03773). Replaces standard autoregressive decoding with an iterative loop that generates reasoning traces, summarizes them, and conditions the next turn on the summary. Decouples effective reasoning horizon from per-step context length.

### Knowledge Distillation

Online distillation from SmolLM2-135M teacher: pretrained embedding transfer at init + soft logit KL divergence during training. Teacher freed at 60% training to reclaim VRAM.

### RL Training (MGPO)

MaxEnt-Guided Policy Optimization adapted from VibeThinker-3B for single-GPU 8GB constraint. On-policy generation, group-relative advantage, prompt difficulty weighting.

## Research Status

> **Phase: Active training on RTX 3070 8GB.** The PyTorch implementation — model, training stack, inference, evaluation — is production-quality. Training runs on a single RTX 3070 with ~74M parameters, 3B tokens, bf16 precision, Muon+AdamW optimizer, and knowledge distillation.

### What Exists

- Full PyTorch library (`src/hagi/`): GDR, HRM, HDIM, MSA, MoE, CAST, NARS, reasoning cache
- Training stack (`scripts/train.py`, `hagi/train/`): 1038-line canonical script with sequential cycling, VRAM probe, prefix LM, distillation, full resume
- Inference stack (`hagi/inference/`): chat REPL, streaming generation, LoRA adapters, online learning, reasoning cache
- Evaluation (`hagi/eval/`): lm-eval-harness adapter, golden benchmarks, intelligence-density metrics
- RL training (`scripts/train_rl.py`, `hagi/train/rl_loop.py`): MGPO loop with reward shaping
- Config (`configs/rtx3070_canonical.yaml`): 1138-line single source of truth with inline rationale for every value

## Key Design Decisions

1. **PyTorch only.** Hypothesis validation requires fast iteration. No Rust/CUDA port until architecture is validated.

2. **`Cl(3,0,0)` (8 blades).** Pragmatic: 64 multiplications per geometric product. Large enough for 4 grades, small enough for negligible compute overhead.

3. **RTX 3070 8GB target.** Every architectural and training decision is constrained by 8GB VRAM. Hidden size 576 (not 768), 2+7+2 layers (not 4+4+4), batch 10, gradient checkpointing, fused CE.

4. **Muon+AdamW hybrid.** Muon (Newton-Schulz orthogonalization) for 2D weight matrices, AdamW for embeddings/norms/gates. Scale-aware weight decay (0.5) bounds weight norms structurally.

5. **manual_bf16 precision.** Model cast to bf16, no autocast. Grads accumulated in fp32. NOT bf16-autocast (OOM'd on 8GB).

6. **Composite loss.** L_CE + L_iso (HDIM invariant alignment) + L_moe (load balance) + L_msa_lb (MSA router load balance) + L_gdr_router (GDR capacity router). Warmup ramps auxiliary terms from 0.

7. **Sequential cycling curriculum.** Datasets iterated in difficulty order (easy coherent text first, hard web+edu last) for N cycles before shuffling.

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Unique parameters | ~74.2M |
| Effective reasoning passes | 14 (7 layers x 2 L-cycles) |
| Hidden size | 576 |
| Attention | GQA (8 query heads, 4 KV heads, head_dim 72) |
| MLP | MoE SwiGLU (4 experts, top-1, 384 intermediate) + MoD skip |
| Position encoding | RoPE (theta 500000, max 4096) |
| Normalization | RMSNorm (pre-norm, fp32 variance) |
| Context length | 1024 (training), 4096 (inference) |
| Vocabulary | 49,152 (SmolLM2 BPE) |
| Clifford algebra | `Cl(3,0,0)`, 8 blades |
| Grade allocation | 64 scalar + 96 vector + 96 bivector + 64 trivector + 256 residual = 576 |
| Training precision | manual_bf16 (model bf16, grads fp32) |
| Attention precision | fp16 (QKV cast for softmax resolution) |
| KV cache | INT8 quantized at inference |
| Optimizer | Muon (2D weights, wd 0.5) + AdamW (1D/embed, wd 0.1) |
| Teacher | SmolLM2-135M (freed at 60% training) |

## VRAM Scaling Projections

| VRAM | Hidden | Batch | L-cycles | Passes | Experts | Est. Params |
|------|--------|-------|----------|--------|---------|-------------|
| 7GB | 576 | 10 | 2 | 14 | 4 | 74M |
| 12GB | 768 | 16 | 2 | 14 | 4 | 131M |
| 16GB | 768 | 24 | 2 | 14 | 6 | 139M |
| 24GB | 1024 | 32 | 3 | 21 | 8 | 235M |
| 48GB | 1536 | 64 | 3 | 21 | 8 | 528M |

## Project Structure

```
HAGI/
|-- src/hagi/              # Library (editable-installed)
|   |-- model/             # Architecture: hagi, gdr, hrm_full, hdim_full, msa, moe, cast, clifford, transformer, kv_cache
|   |-- train/             # loop, optim, checkpoint, config, distillation, rl_loop, rewards
|   |-- data/              # memmap datasets, sequential cycling, SFT, prefix-LM, tokenizer
|   |-- eval/              # evaluate, golden, lm_eval_wrapper, cli
|   |-- inference/         # chat, generate, lora, online, reasoning_cache
|   |-- nars/              # NARS controllers (adapters, bag, budget, truth)
|   |-- losses.py          # CE, fused CE, composite loss
|   `-- utils/             # env, misc
|-- scripts/               # Standalone runners
|   |-- train.py           # Canonical training (1038 lines)
|   |-- chat.py            # Interactive chat REPL
|   |-- train_rl.py        # RL training (MGPO)
|   |-- download_data.py   # Data download + tokenization
|   `-- profile_steps.py   # Synthetic GPU-resident profiling
|-- configs/
|   `-- rtx3070_canonical.yaml  # Single source of truth (1138 lines)
|-- docs/                  # Documentation
|-- data/v4_3b/            # Memmap-packed training data (gitignored)
|-- checkpoints/rtx3070/   # Training checkpoints (gitignored)
|-- torch-compile.bat      # MSVC wrapper for torch.compile on Windows
`-- AGENTS.md              # AI agent instructions (gitignored)
```

## Getting Started

### Prerequisites

- Python 3.13+
- PyTorch 2.0+ with CUDA support
- RTX 3070 8GB (or adjust config for your GPU)

### Setup

```bash
git clone https://github.com/ShmidtS/HAGI.git
cd HAGI
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### Data Download

```bash
# Needs HF_TOKEN in .env
python scripts/download_data.py
```

### Train

```bash
# Canonical training (full-featured)
python scripts/train.py --config configs/rtx3070_canonical.yaml

# Resume from checkpoint
python scripts/train.py --resume checkpoints/rtx3070/step-00039000.pt

# Dry run (build model + one batch, report VRAM, exit)
python scripts/train.py --dry-run

# With torch.compile on Windows
torch-compile.bat scripts/train.py --config configs/rtx3070_canonical.yaml
```

### Chat

```bash
python scripts/chat.py --config configs/rtx3070_canonical.yaml --checkpoint checkpoints/rtx3070
```

### Evaluate

```bash
hagi-eval --checkpoint checkpoints/rtx3070/step-00039000.pt --golden
python -m hagi.eval.evaluate --ckpt checkpoints/rtx3070/step-00039000.pt --benchmarks gsm8k,arc_challenge
```

### Lint / Typecheck

```bash
ruff check .
basedpyright src/hagi
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Detailed architecture specification |
| [docs/TRAINING.md](docs/TRAINING.md) | Training stack, optimizer, loss, workflow |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Config reference for rtx3070_canonical.yaml |
| [docs/INFERENCE.md](docs/INFERENCE.md) | Inference, chat, generation, LoRA, reasoning cache |
| [docs/RESEARCH.md](docs/RESEARCH.md) | Research background and literature review |
| [docs/MILESTONES.md](docs/MILESTONES.md) | Staged roadmap with gates and stop conditions |
| [docs/NARS.md](docs/NARS.md) | Non-Axiomatic Reasoning System documentation |
| [docs/ARCHITECTURE_DIAGRAMS.md](docs/ARCHITECTURE_DIAGRAMS.md) | Mermaid architecture diagrams |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributing guide |
| [CHANGELOG.md](CHANGELOG.md) | Changelog |

## License

Apache 2.0
