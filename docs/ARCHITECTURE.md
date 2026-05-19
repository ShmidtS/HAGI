# HAGI Architecture

## Overview

HAGI (Hypercomplex Artificial General Intelligence) is a unified deep learning system that merges two architectural paradigms:

- **HRM (Hierarchical Recurrent Model)**: A two-level recurrent Transformer where a fast L-module performs local token-level refinement and a slow H-module integrates global context. This creates a compute-efficient backbone that reuses parameters across depth through recurrence.
- **HDIM (Hypercomplex Domain Isomorphism Machine)**: A structural reasoning layer that operates in Clifford algebra, extracting domain-invariant encodings via rotor sandwich products and enabling cross-domain knowledge transfer.

The system is implemented in Rust with GPU kernels compiled through [cuda-oxide](https://github.com/NVlabs/cuda-oxide), an experimental Rust-to-PTX compiler that enables single-source GPU development without DSLs or FFI bindings.

---

## System Architecture

```
Input Tokens
    |
    v
Token Embeddings (hrm-model)
    |
    v
+----------------------------------+
|  HRM Recurrent Backbone          |
|                                  |
|  for h in 0..H_cycles:           |
|    for l in 0..L_cycles:         |
|      L-Transformer (hrm-model)   |  <-- fast local refinement
|    H-Transformer (hrm-model)     |  <-- slow global integration
|    |                             |
|    v                             |
|  HDIM Structural Layer           |
|    - Hidden -> Multivector       |
|    - Rotor Sandwich Extract      |  <-- domain-invariant encoding
|    - Structural MoE (moe)        |  <-- level-specialized experts
|    - Titans Memory (memory)      |  <-- adaptive long-term context
|    - Gated Fusion                |
|    |                             |
|    v                             |
|  (loop to next H-cycle)          |
+----------------------------------+
    |
    v
LM Head -> Logits -> CE Loss
    |
    v
Composite Loss (losses)
    - CE (next-token prediction)
    - Reconstruction
    - Isomorphism
    - InfoNCE / SupCon (pairs/triplets)
    - MoE routing / Z / expert orthogonalization
    - Memory consistency
```

---

## Crate Organization

The workspace is organized into 10 crates, each with a single responsibility:

| Crate | Purpose | Key Types |
|---|---|---|
| `core-types` | Zero-dependency metadata types | `AlgebraSignature`, `Shape`, `DType`, IDs, `PackedSequenceLayout` |
| `tensor-runtime` | Tensor handles + backend trait | `Tensor<T>`, `TensorBackend`, `CpuBackend` |
| `clifford-core` | Clifford algebra primitives | `CliffordAlgebra`, `Multivector`, `Rotor`, `Invariant`, `ProductTable` |
| `hrm-model` | Hierarchical recurrent backbone | `TransformerBlock`, `TransformerStack`, `HrmBackbone`, `LmHead` |
| `hdim-model` | Structural reasoning layer | `HiddenToMultivector`, `InvariantExtractor`, `StructuralFusion` |
| `moe` | Mixture-of-Experts (future) | `Router`, `Expert`, dispatch/combine |
| `memory` | Titans/TTT adaptive memory (future) | `AdaptiveMemory`, `SurpriseUpdater` |
| `data` | Data loading + synthetic pairs | `ToyDataset`, `PrefixLmPacker`, `MultipackScheduler` |
| `losses` | Composite objective functions | `CrossEntropyLoss`, `IsoLoss`, `ContrastiveLoss` |
| `cuda-kernels` | cuda-oxide GPU kernels | `#[kernel]` functions for attention, Clifford ops, MoE |
| `config` | Typed configuration system | `ModelConfig`, `HrmConfig`, `HdimConfig` |

---

## Data Flow

### Training Forward Pass

```
1. data: PrefixLM packer splits examples into prefix + causal regions
2. data: Multipack scheduler groups variable-length sequences
3. hrm-model: Token embeddings -> initial z_H
4. hrm-model: z_L initialized from learned parameter
5. LOOP H-cycles (outer):
     LOOP L-cycles (inner):
       L-Transformer: z_L = f_L(z_L, z_H_injected)
     H-Transformer: z_H = f_H(z_H, z_L_injected)
     hdim-model: mv = project_hidden_to_multivector(z_H)
     hdim-model: inv = rotor_sandwich_extract(mv, source_rotor)
     hdim-model: z_H = gated_fusion(z_H, inv, memory_read)
     memory: update_bank(z_H, inv, surprise)
6. hrm-model: LM Head projects z_H -> logits
7. losses: L_total = L_ce + λ_recon·L_recon + λ_iso·L_iso + λ_pair·L_pair + ...
```

### Inference Pass

Same forward path, but:
- Memory updates are **enabled by default** (configurable read-only mode for reproducibility)
- KV cache is persistent across tokens
- Structural layer can run every H-cycle or at a lower frequency (configurable)
- Domain transfer can be forced, inferred, or disabled per query

---

## Key Design Decisions

### 1. H-Level Structural Layer Only (v1)

Full HDIM processing (multivector projection, rotor sandwich, MoE, memory) runs **after each H-module update**, not during every L-cycle.

**Rationale**: Lower compute cost, cleaner abstraction (H-state is the global controller), easier gradient scheduling. L-cycle Clifford enrichment is planned for v2 once the H-level path validates.

### 2. Fixed Compile-Time Algebra with Learnable Internal Structure

The algebra dimension is fixed at compile time (e.g., `Cl(8,0)` or `Cl(16,0)`) via Rust const generics to keep GPU memory layouts stable and kernel coalescing predictable. Within this fixed space, blade coefficients, rotor parameters, and grade-wise scale factors are all trainable.

**Rationale**: Dynamic blade counts inside hot kernels would fragment memory coalescing and complicate PTX generation. Fixed dimension + learnable weights = structured sparsity.

### 3. Gated Residual Structural Fusion

Structural features from HDIM are fused back into H hidden states via:

```
fused_h = h + sigmoid(gate(h, inv_features, memory_read)) * W_structural(structural_features)
```

**Rationale**: Additive residual is too unconstrained; cross-attention is too expensive. A gate lets the HRM backbone suppress unstable structural signals during early training.

### 4. Pure Rust / cuda-oxide (No PyTorch Adapters)

No transitional tch-rs or Candle dependencies. Parity validation against Python HRM-Text is performed via exported golden-output fixtures, not adapter-based inference.

**Rationale**: PyTorch semantics create a permanent impedance mismatch with Rust ownership, async scheduling, and custom sharding. The user explicitly wants a clean Rust/cuda-oxide stack.

### 5. Explicit Memory Update Policy

Titans/TTT memory updates during inference are ON by default but explicitly configurable. Every checkpoint serializes the full memory bank state.

**Rationale**: Adaptive memory is the core value proposition of Titans/TTT; disabling it by default would hide the system's key differentiator. At the same time, reproducibility requires explicit policy and state capture.

---

## Clifford Algebra in HAGI

### Multivector Representation

A multivector in `Cl(p,q,r)` is a vector of coefficients over basis blades:

```
G = g_0               (scalar)
  + g_1 e_1 + g_2 e_2 + ... + g_n e_n     (vectors)
  + g_12 e_12 + g_13 e_13 + ...           (bivectors)
  + ... + g_12...n e_12...n               (pseudoscalar)
```

Total blade count = `2^(p+q+r)`.

### Geometric Product

The fundamental operation of Clifford algebra. For two multivectors `A` and `B`:

```
(A ⊗ B)_c = Σ_{i,j: e_i * e_j = ±e_c} sign * a_i * b_j
```

This single operation simultaneously encodes:
- **Inner product** (scalar part = dot product / similarity)
- **Outer product** (higher-grade parts = oriented relational structure)

### Rotor Sandwich (Invariant Extraction)

For a domain with learned rotor `R`:

```
U_inv = R^{-1} ⊗ G ⊗ R        (removes domain orientation)
G_target = R_target ⊗ U_inv ⊗ R_target^{-1}   (re-applies target orientation)
```

The invariant `U_inv` is the structural essence of the representation, stripped of domain-specific vocabulary/format fingerprint.

---

## cuda-oxide Integration Strategy

### GPU-Side Kernels (cuda-oxide)

| Kernel Family | Priority | Complexity |
|---|---|---|
| Token embedding lookup | High | Low |
| RoPE | High | Low |
| RMSNorm / LayerNorm | High | Low |
| SwiGLU MLP | High | Medium |
| PrefixLM FlashAttention | Critical | High |
| Hidden -> Multivector projection | Critical | Medium |
| Geometric product | Critical | Medium |
| Rotor sandwich (extract/transfer) | Critical | Medium |
| MoE router / dispatch / combine | Medium | High |
| Titans memory lookup / update | Medium | Medium |
| CE / contrastive loss reductions | Medium | Low |

### Host-Side Responsibilities

- Configuration loading and validation
- Dataset loading, tokenization, PrefixLM packing, multipack scheduling
- Training loop orchestration
- Distributed process control (future)
- Checkpointing (model weights + optimizer state + memory bank + domain rotors)
- Metrics and logging
- Kernel selection and autotune policy

### Fallback Policy

Every cuda-oxide kernel family has four levels:
1. **CPU/reference Rust** — correctness baseline
2. **Simple cuda-oxide kernel** — GPU correctness validation
3. **Optimized cuda-oxide kernel** — production performance
4. **Transitional external backend** — only if parity requires comparison

This is mandatory because cuda-oxide is experimental and APIs may change.

---

## Training Architecture

### Composite Loss Schedule

Losses are ramped progressively to avoid overwhelming CE with structural signals:

```
Phase 1 (warmup):      CE only
Phase 2:               CE + reconstruction + rotor norm regularization
Phase 3:               + isomorphism + contrastive (pairs/triplets)
Phase 4:               + MoE orthogonalization + memory losses
```

Each phase is configurable, not hard-coded.

### Data Pipeline

1. **Raw text** -> tokenization
2. **PrefixLM split** -> prefix (bidirectional) + response (causal)
3. **Synthetic pair generation** (in-domain augmentation + cross-domain transforms)
4. **Multipack greedy/LPT bin-packing** -> `PackedSequenceLayout`
5. **Batch collation** -> `MultipackBatch` with `domain_pairs` metadata

### Toy Configuration (Initial Target)

```
total_layers: 8           (4 H + 4 L)
hidden_size: 256
num_heads: 4
h_cycles: 1
l_cycles: 2
vocab_size: 32000
algebra: Cl(8,0)          (256 blades)
structural_heads: 4
blade_count_per_head: 256
expert_count: 4
memory_bank_size: 256
```

---

## Milestones

| Milestone | Scope | Acceptance |
|---|---|---|
| 0 | Reference contracts, Clifford identities, config validation | `cargo test` passes |
| 1 | Single-GPU forward parity (CPU) | Rust forward matches Python fixtures within tolerance |
| 2 | CE + composite loss training step | One backward pass works, checkpoint save/load |
| 3 | MoE + Titans memory | Routing stable, memory update reproducible |
| 4 | cuda-oxide kernel optimization | GPU kernels match reference, performance baselines |
| 5 | Distributed training | Multi-GPU, sharded parameters, distributed checkpoints |

---

## References

- [HRM-Text](https://github.com/sapientinc/HRM-Text) — Hierarchical recurrent language model backbone
- [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide) — Rust-to-CUDA compiler
- [WSL2 Setup](WSL2-SETUP.md) — platform installation guide
