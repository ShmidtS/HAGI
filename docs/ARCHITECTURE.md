# HAGI Architecture

HAGI is a small language model built around one novel mechanism: **Grade-Decomposed Recurrence (GDR)**. This document specifies the architecture in detail.

For research background and evidence, see [RESEARCH.md](RESEARCH.md). For the staged build plan, see [MILESTONES.md](MILESTONES.md).

---

## Design Goal

Maximize **intelligence density** — reasoning capability per parameter — in a model small enough to run locally. Not to compete with frontier LLMs, but to test whether geometric structure in the recurrence representation improves reasoning.

---

## The Central Hypothesis

Recurrent-depth transformers iterate a shared block over a **flat** hidden vector. Empirically (Huginn), gains plateau after ~8-16 iterations: every dimension of the flat vector converges at roughly the same rate, so additional iterations stop adding information.

HAGI decomposes the hidden state into **Clifford grades** with distinct semantics and distinct update dynamics:

| Grade | Clifford object | Dims | Semantic role | Update rate |
|-------|----------------|------|---------------|-------------|
| 0 | scalar | 64 | confidence / resolution | slow (momentum α=0.9) |
| 1 | vectors | 192 | entity / concept representation | medium (α=0.5) |
| 2 | bivectors | 192 | relations between entities | fast (full update) |
| 3 | trivector | 64 | higher-order structure | fast (full update) |
| — | residual | 256 | unconstrained channel | standard |

The geometric product of `Cl(3,0,0)` mixes grades by its algebraic definition (`vector × vector → scalar + bivector`), so entity-level reasoning automatically generates relational and confidence signals. The hypothesis: this lets useful reasoning continue past the flat-recurrence plateau because relational (bivector) components keep evolving while confidence (scalar) components stabilize.

---

## System Pipeline

```
Input Tokens + Position IDs
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Token Embedding (32K → 768) + RoPE              │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  PERCEPTION — Layers 1-4 (unique params)         │
│  Per block: RMSNorm → GQA → RMSNorm → SwiGLU     │
│  Output: contextual token representations         │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  REASONING CORE — Layers 5-8 (shared, looped N×) │
│                                                  │
│  for i in 1..N:                                  │
│    1. decompose(h) → grades + residual           │
│    2. grade update (per-grade MLP + momentum)    │
│    3. geometric_product cross-grade interaction  │
│    4. recompose → h                              │
│    5. RMSNorm → GQA → RMSNorm → SwiGLU           │
│    6. h += iteration_embedding[i]                 │
│    7. (optional) halt if P(halt) > threshold     │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  EXPRESSION — Layers 9-12 (unique params)        │
│  Per block: RMSNorm → GQA → RMSNorm → SwiGLU     │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  RMSNorm → LM Head (768 → 32K, weight-tied)      │
└──────────────────────────────────────────────────┘
        │
        ▼
   Logits → Cross-Entropy Loss (+ optional L_iso)
```

---

## Grade-Decomposed Recurrence — Detail

### Decomposition

The 768-dim hidden state is split into fixed contiguous slices:

```
h[..., 0:64]      → scalar grade    (S=64)
h[..., 64:256]    → vector grade    (V=192)
h[..., 256:448]   → bivector grade  (B=192)
h[..., 448:512]   → trivector grade (T=64)
h[..., 512:768]   → residual        (R=256)
```

The vector/bivector slices are reshaped so the `Cl(3,0,0)` 8-blade structure applies per structural head. With 192 dims and 8 blades, that is 24 structural heads for vectors (similarly for bivectors).

### Per-Grade Update

Each grade has its own update MLP. The momentum blend controls how fast each grade changes per iteration:

```python
def grade_update(scalar, vector, bivector, trivector):
    ctx = concat(scalar, vector, bivector, trivector)
    scalar_new    = 0.9 * scalar    + 0.1 * mlp_scalar(ctx)
    vector_new    = 0.5 * vector    + 0.5 * mlp_vector(ctx)
    bivector_new  =                        mlp_bivector(ctx)   # no momentum
    trivector_new =                        mlp_trivector(ctx)  # no momentum
    return scalar_new, vector_new, bivector_new, trivector_new
```

The momentum coefficients are the inductive bias under test (Stage 2 ablation). They encode: "overall assessment changes slowly; relational reasoning changes fast."

### Geometric Interaction

Cross-grade mixing via the `Cl(3,0,0)` geometric product:

```python
geo = clifford_geometric_product(vector_new, vector_new)  # 8-blade product
scalar_new   += gate_0 * extract_grade(geo, 0)
bivector_new += gate_2 * extract_grade(geo, 2)
```

Gates are learned scalars (sigmoid), allowing the model to control how much geometric signal feeds each grade.

### Recomposition

Grades and residual are concatenated back into a 768-dim vector, then passed through a standard transformer block (RMSNorm → GQA → RMSNorm → SwiGLU). The iteration embedding is added so the block knows which reasoning step it is on.

---

## Canonical Configuration

| Component | Parameter | Value |
|-----------|-----------|-------|
| Model | unique params | ~115M |
| Model | effective depth (N=3) | 20 layers |
| Model | hidden_size | 768 |
| Model | vocab_size | 32000 |
| Model | context_length | 4096 |
| Attention | num_query_heads | 12 |
| Attention | num_kv_heads | 4 (GQA) |
| Attention | head_dim | 64 |
| Attention | rope_theta | 10000 |
| MLP | type | SwiGLU |
| MLP | intermediate_size | 2048 |
| Norm | type | RMSNorm (pre-norm) |
| Layers | perception | 1-4 (unique) |
| Layers | reasoning core | 5-8 (shared, looped) |
| Layers | expression | 9-12 (unique) |
| Recurrence | loop_count N | 3 (configurable 1-5) |
| Recurrence | iteration_embedding | learned, added per loop |
| Recurrence | halting | optional (PonderNet-style) |
| Clifford | signature | `Cl(3,0,0)` |
| Clifford | blade_count | 8 |
| Grades | scalar / vector / bivector / trivector / residual | 64 / 192 / 192 / 64 / 256 |
| Grades | momentum (scalar / vector) | 0.9 / 0.5 |
| Training | precision | bf16 |
| Deployment | quantization | Q4_K_M (~65MB) |

---

## Loss

```
L_total = L_CE + λ_iso · L_iso
```

- **L_CE** — response-only cross-entropy. Prefix tokens masked.
- **L_iso** — optional isomorphic consistency on grade representations across loop iterations, annealed from 0 over the first 20% of training.

`λ_iso` starts at 0 (model learns base language first), then ramps linearly to target.

---

## Why Three Phases (Perception / Reasoning / Expression)

Separating perception and expression from the looped core is deliberate:

- **Perception layers** map tokens to representations once. No need to repeat this per loop.
- **Reasoning core** is the only part that benefits from iteration. Looping it concentrates compute where reasoning happens.
- **Expression layers** decode the refined representation once.

This mirrors Huginn's Prelude/Loop/Coda but places the Clifford grade structure inside the loop, which Huginn does not.

---

## Ablation Models

The architecture is validated by four models with identical training:

| Model | Perception | Core | Expression | Clifford |
|-------|-----------|------|-----------|----------|
| A (baseline) | L1-4 | L5-8 (no loop) | L9-12 | none |
| B (loop) | L1-4 | L5-8 looped 3× | L9-12 | none |
| C (HDIM) | L1-4 | L5-8 (no loop) | L9-12 | bolted-on projection |
| D (GDR) | L1-4 | L5-8 looped 3× | L9-12 | grade-decomposed in loop |

Critical comparison: **B vs D** isolates the contribution of grade decomposition to recurrence.

---

## Implementation Layers

| Layer | Technology | Status | Stage |
|-------|-----------|--------|-------|
| Prototype | PyTorch | To build | 0-4 |
| Production | Rust (`crates/`) | Scaffolded | 5 |
| GPU kernels | CUDA (`cudarc`/cuda-oxide) | Stub | 5 |
| Verification | Lean4 (`formalization/`) | Proofs exist | 6 |

The PyTorch prototype is the source of truth during research. The Rust port happens only after Stage 2 validates the architecture.

---

## Relationship to Prior HAGI Design

Earlier HAGI documents described HRM (separate H/L transformer stacks), HDIM (bolted-on Clifford projection), MSA (sparse memory), MoE, and Titans memory stacked together. That design was over-scoped — five unproven mechanisms with no baseline.

This architecture replaces it:
- **HRM's H/L distinction** → achieved by grade momentum (slow scalar = "H", fast bivector = "L") within a single shared block. No architectural duplication.
- **HDIM's bolted-on Clifford** → Clifford structure moved *inside* the recurrence (GDR). It is now integral, not additive.
- **MSA / MoE / Titans** → deferred to post-validation stages. Removed from the core experiment to eliminate confounds.

The Rust crates and Lean4 proofs from the prior design are retained and will be refactored to match GDR during Stage 5-6.

---

## References

- [RESEARCH.md](RESEARCH.md) — literature review and evidence classification
- [MILESTONES.md](MILESTONES.md) — staged build plan with gates and pivot conditions
- [implementation_plan.md](implementation_plan.md) — crate-level implementation detail (Rust, Stage 5+)
