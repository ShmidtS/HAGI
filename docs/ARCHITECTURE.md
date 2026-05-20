# HAGI Architecture

HAGI integrates three subsystems with exact data-flow boundaries specified by a Lean4 contract layer and scaffolded in Rust crates per milestone. Verification is theorem-level for formalized invariants; runtime parity is a per-milestone acceptance gate, not yet completed.

- **HRM** — Hierarchical Recurrent Model. Maintains `z_H` (strategic) and `z_L` (tactical) hidden states across `H_cycles` × `L_cycles`. Enforces PrefixLM mask legality and packed-sequence partition invariants.
- **HDIM** — Hypercomplex Domain Isomorphism Machine. Projects HRM hidden states into Clifford multivector space, extracts domain-invariant structure via rotor sandwich, performs cross-domain transfer, and fuses back through gated residual.
- **MSA** — Memory Sparse Attention. Provides long-context memory slots as additional HDIM domains. Enforces document-wise RoPE separation, append-only K/V cache, and routing safety (all selected slot IDs exist).

All three subsystems interact exclusively through the **Tensor Runtime** boundary (shape + dtype + layout + backend dispatch). The **Lean4 Verification Layer** defines the formal contracts each Rust crate must satisfy. GPU kernels are compiled through [cuda-oxide](https://github.com/NVlabs/cuda-oxide) as an optional backend; CPU reference is always available.

For milestone-driven implementation path see [`implementation_plan.md`](implementation_plan.md).
For algorithmic complexity, resource estimates, and realizability verification see [`realizability_verification.md`](realizability_verification.md).
For visual data-flow diagrams see [`architecture_diagrams.md`](architecture_diagrams.md).

---

## System Architecture

```
Input Tokens + Position IDs + PrefixLM Mask + Packed Partition
    |
    v
Token Embeddings (hrm-model)
    |
    v
+----------------------------------+
|  HRM Recurrent Backbone          |
|                                  |
|  for h in 1..H_cycles:           |
|    for l in 1..L_cycles:         |
|      L-Transformer (hrm-model)   |  <-- fast local refinement, PrefixLM mask
|    H-Transition: z_H = f(z_H, z_L)|  <-- strategic state update
|    L-Reset: z_L = g(z_H)         |  <-- tactical state reset
|    |                             |
|    v                             |
|  HDIM Structural Layer           |
|    - Hidden -> Multivector       |
|    - Rotor Sandwich Extract      |  <-- invariant U = R_src^-1 * G * R_src
|    - Domain Transfer             |  <-- G_target = R_target * U * R_target^-1
|    - Gated Fusion                |  <-- residual fusion back to hidden
|    |                             |
|    v                             |
|  MSA Sparse Memory Router        |
|    - Routing key = invariant U_q |
|    - Top-k slot selection        |  <-- Clifford scalar product scoring
|    - Document-wise RoPE + K/V    |
|    - Sparse attention output     |  <-- fed back into HRM
|    |                             |
|    v                             |
|  (loop to next H-cycle)          |
+----------------------------------+
    |
    v
LM Head -> Logits
    |
    v
Composite Loss (losses)
    - L_CE: response-only cross-entropy
    - L_aux: supervised contrastive router alignment
    - L_iso: isomorphic transfer fidelity
```

---

## Training and Evaluation Pipeline

### Data Input Contract

Training consumes a tokenized and sampled corpus directory equivalent to HRM-Text `/dev/shm/sampled`. Each shard contains token IDs, document IDs, sequence lengths, PrefixLM split points, and optional domain labels. The sampler preserves the configured stratification keys when building packed batches; no stratum may be dropped unless its shard count is zero in the input manifest.

### Distributed Launcher Model

Rust training is launched through the training binary, for example:

```bash
cargo run --bin hagi-train -- --config configs/train.toml
```

Multi-process coordination owns rank assignment, checkpoint ownership, metric aggregation, and shutdown. Worker processes run the same forward/backward graph and synchronize gradients through the configured distributed backend before optimizer step.

### Checkpoint Layout

Each epoch writes a separate checkpoint directory:

```text
checkpoints/epoch-000001/
  model.safetensors
  optimizer.bin
  domain_rotors.safetensors
  msa_slots.manifest.json
  config.toml
  metrics.json
```

The checkpoint contains model weights, optimizer state, HDIM domain rotors, MSA slot metadata, and a config snapshot used to reproduce the run.

### Evaluation Harness

Evaluation uses benchmark subset selection through `run_only`, for example `run_only=[MATH,DROP,ARC,MMLU]`. The harness loads the latest complete checkpoint, runs each selected benchmark, and records per-benchmark metrics under the epoch directory. GPU memory is budgeted from model weights, optimizer state when training/eval share a process, activation cache, and selected MSA K/V pages. If allocation fails, the harness retries with a smaller evaluation batch size before reporting out-of-memory.

### HF/Transformers Export

HAGI does not use PyTorch adapters in the runtime path. HF/Transformers export is a serialization boundary for checkpoint inspection and downstream loading. Config fields map as follows:

| HF/Transformers field | HAGI source |
|---|---|
| `hidden_size` | HRM hidden width |
| `num_hidden_layers` | HRM `n_layers` |
| `num_attention_heads` | HRM `num_heads` |
| `H_cycles` | HRM outer recurrence count |
| `L_cycles` | HRM inner recurrence count |
| `L_bp_steps` | HRM truncated backpropagation steps |
| `max_position_embeddings` | tokenizer/context configuration |
| `rope_theta` | RoPE configuration |
| `prefix_lm` | PrefixLM mask mode |

---

## Canonical Model Config

| Component | Parameter | Value/Type | Description |
|---|---|---|---|
| HRM | `n_layers` | e.g. `24` | Transformer block count in the recurrent backbone. |
| HRM | `hidden_size` | e.g. `1280` | Hidden-state width for HRM tokens and fused HDIM output. |
| HRM | `num_heads` | e.g. `10` | Attention head count. |
| HRM | `expansion` | e.g. `4` | MLP expansion factor. |
| HRM | `H_cycles` | e.g. `2` | Outer strategic recurrence cycles. |
| HRM | `L_cycles` | e.g. `3` | Inner tactical recurrence cycles per H-cycle. |
| HRM | `bp_warmup_ratio` | e.g. `0.2` | Fraction of training used to warm up recurrence backpropagation depth. |
| HRM | `bp_max_steps` | e.g. `5` | Maximum truncated backpropagation steps through recurrence. |
| HDIM | `clifford_signature` | e.g. `Cl<3,0,0>` | Compile-time Clifford algebra signature for M0-M6. |
| HDIM | `blade_count` | `usize` | `2^(p+q+r)` coefficients per multivector. |
| HDIM | `rotor_count` | `usize` | Number of registered domain rotors. |
| HDIM | `transfer_domains` | list of `DomainId` pairs | Allowed source-target transfer pairs. |
| MSA | `slot_count` | `usize` | Maximum configured sparse memory slots. |
| MSA | `top_k` | `usize` | Number of slots selected per route. |
| MSA | `routing_key_dim` | `usize` | Flattened routing-key width after invariant extraction. |
| MSA | `kv_storage_tier` | GPU routing keys / host content K/V | Routing keys remain GPU-resident; selected content K/V pages are fetched from host memory. |
| Tensor Runtime | `default_dtype` | `DType` | Default tensor dtype for model parameters and activations. |
| Tensor Runtime | `default_layout` | `Layout` | Default physical tensor layout. |
| Tensor Runtime | `backend_parity_tolerance` | e.g. `1e-4` | Maximum absolute error for CPU/CUDA golden-output parity. |

---

## MSA Lifecycle

### Offline Memory Encoding

The offline encoder chunks source documents, pools per-chunk content keys and values, and builds a routing-key cache `(K̄, V̄, K̄ᵣ)`. `K̄` and `V̄` are content K/V pages. `K̄ᵣ` is the compact routing key derived from the HDIM invariant.

### Online Routing

At inference time, the active hidden state projects to routing query `Qᵣ`. The router scores `Qᵣ` against `K̄ᵣ`, selects top-k slot IDs, and validates each selected slot against the registry before content fetch.

### Sparse Generation

Generation attends only over local context plus the selected sparse memory context. Unselected slots do not enter the attention K/V set.

### Memory Interleave

The interleave loop is:

```text
generate retrieval query -> route top-k slots -> expand context -> generate -> repeat
```

The loop stops when it reaches the configured maximum rounds, retrieval returns no slots, or the confidence threshold is met.

### Storage and Fetch Contract

Routing-key shards stay on GPU for scoring. Content K/V pages remain in host memory until selected. The async fetch contract is: selected slot IDs are issued to the host cache, K/V pages are copied to the target backend stream, and sparse attention waits on the fetch completion event before reading them.

---

## HDIM Public API and Transfer State

### Public API Surface

The HDIM crate exposes `CliffordAlgebra`, `DomainRotationOperator`, `InvariantExtractor`, `sandwich_transfer`, `InvariantIndex`, `HDIMCoreEngineConfig`, and `HDIMCoreEngine`.

### TransferState Fields

`TransferState` records `g_source`, `u_inv`, `u_mem`, `u_route`, `g_target`, `output`, `memory_loss`, `router_state`, and `memory_mode`.

### Domain Rotor Lifecycle

Domain rotors follow a fixed lifecycle: register domain, initialize rotor, normalize after update, checkpoint. Checkpointed rotors are part of the training checkpoint and must be restored before transfer or routing replay.

---

## Formal Contract Map

| Lean theorem / contract | Rust module / function | Test / golden fixture |
|---|---|---|
| `unit_rotor_sandwich_identity` | `clifford-core::Rotor`, `hdim-model::sandwich_transfer` | `golden/hdim/rotor_sandwich_identity.json` |
| `same_rotor_transfer_identity` | `hdim-model::DomainRotationOperator`, `hdim-model::InvariantExtractor` | `golden/hdim/same_rotor_transfer_identity.json` |
| Norm preservation for unit rotor sandwich | `clifford-core::CliffordOps::geometric_product` | `golden/clifford/norm_preservation.json` |
| HRM recurrence depth monotonicity | `hrm-model::HrmBackbone::forward_cycles` | `tests/hrm/recurrent_depth_monotonicity.rs` |
| `RouteWithinSlots` | `msa::SparseRouter::route_top_k`, `msa::SlotRegistry` | `tests/msa/route_within_slots.rs` |
| Tensor shape/dtype/layout preservation | `tensor-runtime::Tensor`, `tensor-runtime::Backend` dispatch | `golden/tensor/shape_dtype_layout_preservation.json` |

---

## Crate Organization

| Crate | Purpose | Key Types |
|---|---|---|
| `core-types` | Zero-dependency metadata types | `Shape`, `Layout`, `DType`, `DomainId`, `CycleId`, `AlgebraSignature` |
| `tensor-runtime` | Tensor handles + backend dispatch | `Tensor<T>`, `Backend` enum (`cpuReference` / `cudaOxide`) |
| `clifford-core` | Clifford algebra primitives | `Multivector`, `Rotor`, `ProductTable`, `CliffordOps` |
| `hrm-model` | Hierarchical recurrent backbone | `TransformerBlock`, `HrmBackbone`, `PrefixLMLegal`, `PackedPartition` |
| `hdim-model` | Structural reasoning layer | `HiddenToMultivector`, `InvariantExtractor`, `GatedFusion` |
| `msa` | Sparse memory routing + attention | `SlotRegistry`, `SparseRouter`, `DocumentWiseRoPE`, `KVCache` |
| `data` | Data loading + synthetic pairs | `ToyDataset`, `PrefixLmPacker` |
| `losses` | Composite objective functions | `ResponseCrossEntropy`, `ContrastiveAuxLoss`, `IsomorphicLoss`, `MagicNormClifford` |
| `cuda-kernels` | cuda-oxide GPU kernels | `#[kernel]` functions for geometric product, rotor sandwich, sparse attention |
| `config` | Typed configuration system | `ModelConfig`, `HrmConfig`, `HdimConfig` |

---

## Data Flow

### Training Forward Pass

```
1. data: PrefixLM packer splits examples into prefix (bidirectional) + response (causal)
2. data: Packed-sequence partition validation (no overlap)
3. hrm-model: Token embeddings -> initial z_H, z_L
4. LOOP H-cycles (outer):
     LOOP L-cycles (inner):
       L-Transformer: z_L = f_L(Embed(tokens) + project_z_L(z_L), mask=prefix_mask)
     H-Transition: z_H = f_H(z_H, z_L)
     L-Reset: z_L = g(z_H)
     hdim-model: G = project_hidden_to_multivector(z_H)
     hdim-model: U = rotor_sandwich_extract(G, source_rotor)
     hdim-model: z_H = gated_fusion(z_H, U)
5. hrm-model: LM Head projects z_H -> logits
6. losses: L_total = L_CE + lambda_aux * L_aux + lambda_iso * L_iso
```

### Inference Pass

Same forward path, but:
- MSA sparse memory router is active: structural invariant `U_q` scores against slot registry, top-k slots selected, K/V retrieved with document-wise RoPE.
- KV cache is append-only and persistent across tokens.
- HDIM layer runs after each H-cycle.
- Backend dispatch routes Clifford ops to `cudaOxide` when available; falls back to `cpuReference`.

---

## Key Design Decisions

### 1. H-Level Structural Layer Only (v1)

Full HDIM processing (multivector projection, rotor sandwich, domain transfer, gated fusion) runs **after each H-module update**, not during every L-cycle.

**Rationale**: Lower compute cost, cleaner abstraction (H-state is the global controller), easier gradient scheduling. L-cycle Clifford enrichment is out of M0-M6 scope. It will be added only after H-level HDIM passes parity tests: fixed-seed forward, gradient check, and loss non-regression on the HRM smoke dataset.

### 2. Fixed Compile-Time Algebra with Learnable Internal Structure

The algebra dimension is fixed at compile time. **M1 starts with hardcoded `Cl<3,0,0>`** (8 blades) to avoid proc-macro / const-generic limits in Rust stable. Within this fixed space, blade coefficients, rotor parameters, and grade-wise scale factors are all trainable.

**Rationale**: Dynamic blade counts inside hot kernels would fragment memory coalescing and complicate PTX generation. Fixed dimension + learnable weights = structured sparsity. M0-M6 support only `Cl<3,0,0>`. Generic `Cl<p,q,r>` requires a new milestone with product-table generation, CPU/CUDA parity tests, and memory-layout benchmarks.

### 3. Gated Residual Structural Fusion

Structural features from HDIM are fused back into H hidden states via:

```
fused_h = h + sigmoid(W_gate * concat(h, flatten(G_target))) * W_structural(flatten(G_target))
```

**Rationale**: Additive residual is too unconstrained; cross-attention is too expensive. A gate lets the HRM backbone suppress unstable structural signals during early training.

### 4. Pure Rust / cuda-oxide (No PyTorch Adapters)

No transitional tch-rs or Candle dependencies. Parity validation against reference implementations is performed via exported golden-output fixtures, not adapter-based inference.

**Rationale**: PyTorch semantics create a permanent impedance mismatch with Rust ownership, async scheduling, and custom sharding. The system is built as a clean Rust/cuda-oxide stack.

### 5. Explicit Memory Update Policy

MSA K/V cache is **append-only** during inference. Old slots remain as prefix; new slots append. Routing safety asserts that every selected `slotId` exists in registry.

**Rationale**: Append-only invariant (`CacheAppendOnly`) is provable in Lean4 and guarantees monotonic memory growth. Routing safety (`RouteWithinSlots`) prevents silent retrieval failures at 100M-token scale.

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

**M1 scope**: `Cl<3,0,0>` => 8 blades. Coefficient array length is enforced at construction time by `bladeCountEq : bladeCount = 2^(p+q+r)` in Lean4 `HDIM.lean`.

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
U = R^{-1} ⊗ G ⊗ R        (removes domain orientation)
G_target = R_target ⊗ U ⊗ R_target^{-1}   (re-applies target orientation)
```

The invariant `U` is the structural essence of the representation, stripped of domain-specific vocabulary/format fingerprint.

**Identity property**: The symbolic rotor sandwich identity is proven in Lean (`unit_rotor_sandwich_identity`, `HDIM.lean:140`). Numeric approximation will be accepted by golden-output tests with tolerance `ε` defined in `TensorRuntime.lean` once the CPU reference and CUDA kernels are both implemented; the bound is not yet derived from ULP operation count.

---

## cuda-oxide Integration Strategy

### GPU-Side Kernels (cuda-oxide)

| Kernel Family | Priority | Complexity | Milestone |
|---|---|---|---|
| Token embedding lookup | High | Low | M2 |
| RoPE | High | Low | M2 |
| RMSNorm / LayerNorm | High | Low | M2 |
| SwiGLU MLP | High | Medium | M2 |
| PrefixLM FlashAttention | Critical | High | M2 |
| Hidden -> Multivector projection | Critical | Medium | M3 |
| Geometric product | Critical | Medium | M1 / M6 |
| Rotor sandwich (extract/transfer) | Critical | Medium | M3 / M6 |
| Sparse attention score + routing | Medium | High | M4 / M6 |
| CE / contrastive loss reductions | Medium | Low | M5 |

### Host-Side Responsibilities

- Configuration loading and validation
- Dataset loading, tokenization, PrefixLM packing
- Training loop orchestration with truncated BPTT (horizon `K << N`)
- Checkpointing (model weights + optimizer state + domain rotors + K/V cache)
- Metrics and logging
- Kernel selection and autotune policy

### Fallback Policy

Every cuda-oxide kernel family has three levels:
1. **CPU/reference Rust** — correctness baseline (M0-M5)
2. **Simple cuda-oxide kernel** — GPU correctness validation (M6)
3. **Optimized cuda-oxide kernel** — production performance (M6+)

CPU reference is the semantic source of truth. The CUDA backend must pass golden-output parity before use. cuda-oxide API breakage blocks only the CUDA backend, not model correctness.

---

## Training Architecture

### Composite Loss

The total loss is a weighted sum of three components:

```
L_total = L_CE + lambda_aux * L_aux + lambda_iso * L_iso
```

- **L_CE**: response-only cross-entropy. Computed only on causal suffix tokens; prefix tokens are masked out.
- **L_aux**: supervised contrastive alignment. Maximizes cosine similarity of structurally matching invariants, minimizes for non-matching pairs.
- **L_iso**: isomorphic transfer fidelity. `L_iso = ||U_source - U_target||^2` penalizes domain-transfer discrepancy.

MagicNorm-Clifford gradient-bound claims are pending M5 backward-pass implementation.

### Data Pipeline

1. **Raw text** -> tokenization
2. **PrefixLM split** -> prefix (bidirectional) + response (causal)
3. **Packed-sequence partition** -> variable-length sequences grouped with no overlap
4. **Batch collation** -> `PackedPartition` with `domain_pairs` metadata

### Toy Configuration (M5 Overfitting Test)

```
total_layers: 6
hidden_size: 512
num_heads: 8
h_cycles: 2
l_cycles: 2
vocab_size: 32000
algebra: Cl<3,0,0>          (8 blades)
structural_heads: 8
blade_count_per_head: 8
msa_slots: 100
```

Resource estimates are pending benchmark harness (M5). Target configuration: 24 layers, 1280 hidden size, 10 heads. See `docs/implementation_plan.md` for per-milestone memory and compute budgets.

---

## Milestones

Milestones are defined in detail in [`implementation_plan.md`](implementation_plan.md). Summary:

| Milestone | Scope | Acceptance |
|---|---|---|
| M0 | Tensor Runtime & Core Types | `cargo test` + `lake build` pass |
| M1 | Clifford Algebra Core (`Cl<3,0,0>`) | Cayley table correct; rotor sandwich ≈ identity |
| M2 | HRM Backbone (forward-only CPU) | Shape preservation; PrefixLM mask unit tests |
| M3 | HDIM Layer (projection → invariant → transfer → fusion) | Round-trip shape unchanged; `same_rotor_transfer_identity` proven |
| M4 | MSA Sparse Memory (CPU reference) | Routing safety; append-only cache; RoPE separation |
| M5 | Composite Loss & Training Loop | Loss finite, grad non-NaN; toy overfitting in 100 steps |
| M6 | CUDA-Oxide Backend Kernels | CPU vs CUDA ≈ ε; CUDA kernel fusion target: benchmark against CPU reference after M6. No validated speedup claim yet. |

**Critical path (training)**: M0 → M1 → M3 → M5.
**Secondary path (inference scaling)**: M0 → M2 → M4 → M6.

M3 and M2 can be developed in parallel after M0. M5 depends on M2 + M3 + M4 interfaces. M6 depends on M1 + M4.

---

## References

- [`implementation_plan.md`](implementation_plan.md) — Milestone-driven implementation path with contracts and stop-conditions.
- [`architecture_diagrams.md`](architecture_diagrams.md) — Mermaid diagrams of system architecture, HRM recurrence, HDIM pipeline, MSA routing, data flow, composite loss, and verification stack.
- [`realizability_verification.md`](realizability_verification.md) — Complexity assessment, resource estimates, risk mitigation, and realizability verdict.
- [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide) — Rust-to-CUDA compiler.
