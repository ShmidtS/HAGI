# HAGI Architecture

HAGI is a small language model built around five novel mechanisms integrated into a single **Perception → Reasoning → Expression** pipeline. This document specifies the architecture in detail.

For research background, see [RESEARCH.md](RESEARCH.md). For the training workflow, see [TRAINING.md](TRAINING.md). For config reference, see [CONFIGURATION.md](CONFIGURATION.md).

---

## Design Goal

Maximize **intelligence density** — reasoning capability per parameter — in a model small enough to run on a single RTX 3070 (8GB VRAM). Not to compete with frontier LLMs, but to test whether geometric structure in the recurrence representation improves reasoning.

---

## The Central Hypothesis

Recurrent-depth transformers iterate a shared block over a **flat** hidden vector. Empirically (Huginn), gains plateau after ~8-16 iterations: every dimension of the flat vector converges at roughly the same rate, so additional iterations stop adding information.

HAGI decomposes the hidden state into **Clifford grades** with distinct semantics and distinct update dynamics:

| Grade | Clifford object | Dims (hidden=576) | Semantic role | Update rate |
|-------|----------------|-------------------|---------------|-------------|
| 0 | scalar | 64 | confidence / resolution | slow (momentum 0.8) |
| 1 | vectors | 96 | entity / concept representation | medium (momentum 0.5) |
| 2 | bivectors | 96 | relations between entities | fast (full update) |
| 3 | trivector | 64 | higher-order structure | fast (full update) |
| -- | residual | 256 | unconstrained channel | standard |

The geometric product of `Cl(3,0,0)` mixes grades by its algebraic definition (`vector x vector -> scalar + bivector`), so entity-level reasoning automatically generates relational and confidence signals. The hypothesis: this lets useful reasoning continue past the flat-recurrence plateau because relational (bivector) components keep evolving while confidence (scalar) components stabilize.

---

## System Pipeline

```
Input Tokens + Position IDs
        |
        v
+----------------------------------------------+
|  Token Embedding (49K -> 576) + RoPE         |
+----------------------------------------------+
        |
        v
+----------------------------------------------+
|  PERCEPTION -- 2 layers (unique params)      |
|  Per block: RMSNorm -> GQA -> RMSNorm -> MoE |
|  Output: contextual token representations    |
+----------------------------------------------+
        |
        v
+----------------------------------------------+
|  REASONING CORE -- 7 layers (HRM-controlled) |
|                                              |
|  for h in 1..H_cycles (default 1):           |
|    for l in 1..L_cycles (default 2):         |
|      1. HDIM: project hidden -> multivector  |
|         extract invariant U = R^-1 * G * R   |
|         transfer domain: G' = R_tgt * U * R  |
|         gated fusion back to hidden          |
|      2. GDR: decompose(h) -> grades + resid  |
|         per-grade update (MLP + momentum)     |
|         geometric_product cross-grade mix    |
|         recompose -> h                       |
|      3. Transformer block:                   |
|         RMSNorm -> GQA -> RMSNorm -> MoE     |
|      4. MSA: slot routing + sparse attention |
|         (memory-aware: read+write per cycle) |
|      5. H/L state transitions                |
|                                              |
|  Parameters shared across all iterations.    |
|  ~74M params; 14 reasoning passes/step.      |
+----------------------------------------------+
        |
        v
+----------------------------------------------+
|  EXPRESSION -- 2 layers (unique params)      |
|  Per block: RMSNorm -> GQA -> RMSNorm -> MoE |
+----------------------------------------------+
        |
        v
+----------------------------------------------+
|  RMSNorm -> LM Head (576 -> 49K, tied)       |
|  or CAST: K=8 token block prediction         |
+----------------------------------------------+
        |
        v
   Logits -> Composite Loss
   L_CE + L_iso + L_moe + L_msa_lb + L_gdr_router
```

---

## Component Specifications

### GDR -- Grade-Decomposed Recurrence

The core novel mechanism. Applied once per reasoning iteration inside the HRM loop.

#### Decomposition

The 576-dim hidden state is split into fixed contiguous slices:

```
h[..., 0:64]      -> scalar grade    (S=64)
h[..., 64:160]    -> vector grade    (V=96)
h[..., 160:256]   -> bivector grade  (B=96)
h[..., 256:320]   -> trivector grade (T=64)
h[..., 320:576]   -> residual        (R=256)
```

The vector slice is reshaped so the `Cl(3,0,0)` 8-blade structure applies per structural head: 96 dims / 8 blades = 12 structural heads.

#### Per-Grade Update

Each grade has its own update MLP. The momentum blend controls how fast each grade changes per iteration:

```python
def grade_update(scalar, vector, bivector, trivector):
    ctx = concat(scalar, vector, bivector, trivector)
    scalar_new    = 0.8 * scalar    + 0.2 * mlp_scalar(ctx)
    vector_new    = 0.5 * vector    + 0.5 * mlp_vector(ctx)
    bivector_new  =                        mlp_bivector(ctx)   # no momentum
    trivector_new =                        mlp_trivector(ctx)  # no momentum
    return scalar_new, vector_new, bivector_new, trivector_new
```

Implementation uses a shared trunk (Linear + SiLU) followed by a single fused head instead of four separate MLPs, cutting eight matmuls down to two.

#### Geometric Interaction

Cross-grade mixing via the `Cl(3,0,0)` geometric product:

```python
geo = geometric_product_self_g02(vector_new)  # 8-blade product
scalar_new   += gate_0 * extract_grade(geo, 0)
bivector_new += gate_2 * extract_grade(geo, 2)
```

Gates are learned scalars (sigmoid), allowing the model to control how much geometric signal feeds each grade.

#### Learnable GDR Capacity Router

An optional MoE-style router gates the per-grade update magnitude so the model self-allocates update energy across grades instead of the fixed 64/96/96/64 split. The grade **dimensions** stay (Clifford needs `vector % 8 == 0`); the router scales magnitude, so geometric product math is unchanged. A Shazeer/Switch load-balance auxiliary loss keeps the gate from collapsing.

Source: `src/hagi/model/gdr.py`

---

### HRM -- Hierarchical Recurrent Model

Two-level reasoning loop with strategic (z_H) and tactical (z_L) recurrent states.

#### States

| State | Dim | Update frequency | Role |
|-------|-----|-----------------|------|
| z_H | 160 | Once per H-cycle | Strategic direction |
| z_L | 160 | Every L-cycle | Tactical execution |

#### Loop Structure

```
for h in 1..H_cycles (default 1):
    for l in 1..L_cycles (default 2):
        x = Embed(tokens) + project_z_L(z_L)
        x = TransformerBlock(x, mask, partition)
        z_L = UpdateL(x, z_L)          # tactical recurrence
    z_H = UpdateH(z_H, z_L)            # strategic update
    z_L = ResetL(z_H)                  # tactical reset
```

#### Memory-Aware HRM

When `hrm_memory_aware=true`, MSA read+write moves **inside** the L-cycle loop. Each reasoning cycle reads the slot registry accumulated by the prior cycle and writes back the refined hidden. This makes the slot registry part of the thinking process (HRM <-> MSA bidirectional) instead of a bolt-on block after reasoning.

#### Stochastic Depth

`hrm_stochastic_depth=0.3` skips L-cycle 1 in 30% of training steps, saving ~15% reasoning compute on average. Acts as regularization (stochastic depth), forcing cycle 0 to be self-sufficient.

#### Progressive Reasoning Budget

`hrm_progressive_start_step=30000` uses 1 L-cycle for the first 30K steps, then full 2 L-cycles. Early training focuses on basic token prediction (doesn't need deep reasoning), saving ~10% total training time.

Source: `src/hagi/model/hrm_full.py`

---

### HDIM -- Hidden-state Decomposed Invariant Module

Cross-domain invariant transfer via Clifford rotor sandwiches.

#### Pipeline

1. **Projection**: `hidden [B,T,576] -> multivector G [B,T,heads,8]` via `HiddenToMultivector`
2. **Invariant extraction**: `U = R_src^-1 * G * R_src` (rotor sandwich)
3. **Domain transfer**: `G_target = R_tgt * U * R_tgt^-1`
4. **Gated fusion**: `hidden += gate * W_fuse(flatten(G_target))`

#### Domain Rotors

Learnable unit even multivectors. Each rotor is a distinct cross-domain invariant-transfer schedule picked per step via LCG (deterministic, no GPU sync). Default: 4 rotors. The rotor schedule is controlled by `rotor_seed` for reproducibility.

#### Delayed HDIM

`hdim_delay_steps > 1` enables delayed aggregation: HDIM results from multiple steps are accumulated before fusion, smoothing the invariant signal.

Source: `src/hagi/model/hdim_full.py`

---

### MSA -- Memory Sparse Attention

Slot-based sparse attention with external memory and HDIM-integrated routing.

#### Slot Registry

Each slot stores:
- `slot_id`: unique identifier
- `domain_id`: HDIM domain assignment
- `routing_key`: Clifford scalar invariant for routing
- `k_cache`, `v_cache`: append-only K/V tensors

Slots are evicted (oldest first) when `msa_slot_count` is exceeded. Default cap: 4096 slots.

#### Routing

1. Project active hidden state to routing query `Q_r` via HDIM invariant extraction
2. Score every slot by dot product `score = Q_r . K_bar_r`
3. Top-k selection (default k=6)
4. Async fetch selected K/V pages
5. Sparse attention over local context + fetched pages

#### Training Behavior

The model-owned registry is cleared every forward pass in training, so the training registry only ever holds the current batch's slots. The larger cap matters at **generation**, where a persistent `SlotRegistry` accumulates across decode steps.

#### Chunk Compression

`msa_chunk_size=4` groups 4 adjacent tokens into one slot via mean-pooling, giving 4x more context at the same slot budget. Effective MSA context = `msa_slot_count * msa_chunk_size = 4096 * 4 = 16384` tokens.

#### Adaptive Top-K

When `msa_adaptive_top_k=true`, top_k is reduced for tokens whose MoE skip-router score is high (trivial tokens get fewer memory slots). Expected ~25% MSA attention reduction with no quality loss.

#### LSH Routing (Disabled)

LSH sublinear routing is implemented but disabled for RTX 3070: the Python loop over queries with per-query GPU sync dominates forward time without changing routing quality. Exact matmul+topk over <=4096 keys is cheaper and exact.

Source: `src/hagi/model/msa.py`

---

### MoE -- Mixture of Experts

SwiGLU expert routing replacing standard SwiGLU in transformer blocks.

| Parameter | Value |
|-----------|-------|
| Experts | 4 |
| Top-k | 1 (Switch Transformer style) |
| Intermediate size | 384 per expert |
| Load-balance aux | 0.01 (Shazeer/Switch) |
| Router temperature | 1.0 |

#### Mixture-of-Depths (MoD)

`moe_mod_skip=true` adds an extra "skip" router slot. Tokens that win the skip slot bypass the experts (residual identity, output 0), saving MLP compute for trivial tokens. The skip slot is excluded from the load-balance aux loss.

Source: `src/hagi/model/moe.py`

---

### CAST -- Clifford Algebra Symbolic Reasoning Tokens

Block-wise generation: each forward pass predicts K=8 tokens via multivector virtual states with geometric product coherence.

#### Forward

```
hidden [B, T, H]
    -> block_proj: Linear(H -> K*H)
    -> reshape to [B, T, K, H]
    -> reshape to multivectors [B, T, K, H//8, 8]
    -> geometric_product(adjacent K positions) -> bivector area
    -> area modulates both neighbour virtual states
    -> flatten back to [B, T, K, H]
```

Each virtual state is decoded through the shared `final_norm + lm_head`, producing K token predictions per position. This reduces sequential forward passes during generation by 8x.

#### Training

Multi-token prediction: position t predicts tokens t+1..t+8. Loss = weighted mean over K positions of fused linear CE. `train_k=3` subsamples: always includes k=0 (next-token), randomly samples 2 from k=1..7. `k_loss_decay=0.5` weights closer predictions higher.

#### Coherence Gate

`gate_init=0.0` means sigmoid(-5) ~ 0.007: coherence disabled at init. Model learns to activate the gate as k>0 positions start predicting useful tokens.

Source: `src/hagi/model/cast.py`

---

### Clifford Algebra Core

`Cl(3,0,0)`: three orthonormal basis vectors e1, e2, e3, each squaring to +1. 8 basis blades indexed by 3-bit bitmask:

```
0b000 = 1            (grade 0, scalar)
0b001 = e1           (grade 1)
0b010 = e2           (grade 1)
0b100 = e3           (grade 1)
0b011 = e1 e2        (grade 2, bivector)
0b101 = e1 e3        (grade 2, bivector)
0b110 = e2 e3        (grade 2, bivector)
0b111 = e1 e2 e3     (grade 3, trivector / pseudoscalar)
```

The geometric product of two basis blades: `result_blade = a XOR b`, `sign = (-1)^(reordering transpositions)`.

The Cayley table is built at import time and checked against the Lean4 formal specification.

Source: `src/hagi/model/clifford.py`

---

### Transformer Block

Standard pre-norm transformer block with optimizations:

| Feature | Implementation |
|---------|---------------|
| Normalization | RMSNorm (fp32 variance computation) |
| Attention | GQA (8 query, 4 KV heads, head_dim 72) |
| Position encoding | RoPE (theta 500000, max 4096) |
| QK-norm | RMSNorm on Q/K after RoPE |
| FFN | MoE SwiGLU or standard SwiGLU |
| Fused QKV | Single matmul for Q, K, V projections |
| Fused gate-up | Single matmul for SwiGLU gate + up |
| fp16 attention | QKV cast to fp16 for softmax (8x better resolution) |
| INT8 KV cache | Per-head fp16 scales, 2x cache reduction |

Source: `src/hagi/model/transformer.py`, `src/hagi/model/kv_cache.py`

---

## Loss

```
L_total = L_CE
        + w_iso   * L_iso       (HDIM domain invariant alignment)
        + w_moe   * L_moe       (MoE load-balance)
        + w_msa_lb * L_msa_lb   (MSA router load-balance)
        + w_gdr_router * L_gdr  (GDR capacity router load-balance)
```

### Component Details

| Loss | Weight | Description |
|------|--------|-------------|
| L_CE | 1.0 | Response-only cross-entropy (fused linear CE, label smoothing 0.05) |
| L_iso | 0.02 | MSE between source/target HDIM invariants |
| L_moe | 0.005 | Shazeer/Switch load-balance on expert routing |
| L_msa_lb | 0.01 | Load-balance on MSA slot routing |
| L_gdr_router | 0.005 | Load-balance on GDR grade-capacity router |
| L_aux | 0.0 | Contrastive auxiliary (DISABLED -- broken labels) |

### Warmup

All auxiliary losses ramp from 0 to target over `warmup_steps` (1000 steps). This lets the model learn base language first, then geometric alignment.

### Fused CE

When `use_fused_ce=true`, `logits` is `None` in training forward; loss computed via `fused_linear_cross_entropy` without materializing `[B, T, V]`. Peak logits memory = `chunk_size * V * dtype_bytes`, never the full tensor.

Source: `src/hagi/losses.py`

---

## Ablation Models

Four models, identical training, architecture-only differences:

| Model | `use_loop` | `use_gdr` | Tests |
|-------|-----------|-----------|-------|
| A (baseline) | false | false | Dense transformer control |
| B (loop) | true | false | Recurrence only |
| C (HDIM) | false | true | Clifford bolted on |
| D (GDR) | true | true | Full HAGI |

Critical comparison: **B vs D** isolates the contribution of grade decomposition to recurrence.

---

## Critical Runtime Invariants

- `use_fused_ce: true` -> `logits` is `None` in training forward; loss computed via fused linear CE. Do NOT access `logits` directly in training code.
- `precision: manual_bf16` -> model cast to bf16, no autocast. Grads accumulated in fp32. NOT bf16-autocast (OOM'd on 8GB).
- Optimizer = `muon_adamw` (Muon for 2D weights via Newton-Schulz + AdamW for rest). NOT plain AdamW.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in `train.py:11`. Windows ignores it -- benign warnings filtered in `__init__.py` and `loop.py`.
- `hagi/__init__.py` filters warnings BEFORE torch import -- import order matters.
- MSA slot registry is cleared every forward pass in training. The registry only persists across decode steps at inference.

---

## Canonical Configuration

| Component | Parameter | Value |
|-----------|-----------|-------|
| Model | unique params | ~74.2M |
| Model | effective reasoning passes | 14 (7 layers x 2 L-cycles) |
| Model | hidden_size | 576 |
| Model | vocab_size | 49152 (SmolLM2 BPE) |
| Model | context_length (training) | 1024 |
| Model | context_length (inference) | 4096 |
| Attention | num_query_heads | 8 |
| Attention | num_kv_heads | 4 (GQA) |
| Attention | head_dim | 72 |
| Attention | rope_theta | 500000 |
| Attention | precision | fp16 (QKV cast) |
| MLP | type | MoE SwiGLU |
| MLP | experts | 4 (top-1, 384 intermediate) |
| MLP | MoD skip | enabled |
| Norm | type | RMSNorm (pre-norm, fp32 variance) |
| Layers | perception | 2 (unique) |
| Layers | reasoning core | 7 (shared, HRM-controlled) |
| Layers | expression | 2 (unique) |
| HRM | H_cycles | 1 |
| HRM | L_cycles | 2 |
| HRM | h_dim / l_dim | 160 / 160 |
| HRM | stochastic_depth | 0.3 |
| HRM | memory_aware | true |
| Clifford | signature | `Cl(3,0,0)` |
| Clifford | blade_count | 8 |
| Grades | scalar / vector / bivector / trivector / residual | 64 / 96 / 96 / 64 / 256 |
| Grades | momentum (scalar / vector) | 0.8 / 0.5 |
| MSA | slot_count | 4096 |
| MSA | top_k | 6 |
| MSA | chunk_size | 4 |
| CAST | block_size | 8 |
| CAST | train_k | 3 |
| Training | precision | manual_bf16 |
| Training | optimizer | muon_adamw |
| Training | muon_lr | 0.02 |
| Training | muon_weight_decay | 0.5 |
| Training | batch_size | 10 |
| Training | grad_accum | 2 |
| Training | schedule | WSD |
| Training | warmup_steps | 1000 |
| Training | train_tokens | 3B |
| Inference | KV cache | INT8 |
| Inference | compile | true (torch.compile) |

---

## References

- [RESEARCH.md](RESEARCH.md) -- literature review and evidence classification
- [CONFIGURATION.md](CONFIGURATION.md) -- config reference with rationale
- [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) -- Mermaid diagrams
- [TRAINING.md](TRAINING.md) -- training stack and workflow
- [INFERENCE.md](INFERENCE.md) -- inference and generation
