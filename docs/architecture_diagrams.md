# HAGI Architecture Diagrams

Diagrams are Mermaid-compatible. Each diagram maps to a subsystem or pipeline stage from `implementation_plan.md`.

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph Input["Input Boundary"]
        TOK[Token IDs + Position IDs]
        MASK[PrefixLM Mask]
        PART[Packed Partition]
    end

    subgraph HRM["HRM — Hierarchical Recurrent Model"]
        direction TB
        EMB[Embedding Layer]
        TBLOCK[TransformerBlock<br/>Self-Attn + MLP]
        HSTATE[HTransition<br/>z_H strategic state]
        LSTATE[LTransition<br/>z_L tactical state]
    end

    subgraph HDIM["HDIM — Domain-Invariant Mapping"]
        direction TB
        PROJ[HiddenToMultivector<br/>[B,T,hidden] -> [B,T,heads,blades]]
        EXTR[InvariantExtractor<br/>U = R^-1 * G * R]
        XFER[DomainTransfer<br/>G_target = R_target * U * R_target^-1]
        FUSE[GatedFusion<br/>gated residual -> hidden]
    end

    subgraph MSA["MSA — Memory Sparse Attention"]
        direction TB
        RKEY[Routing Key Compute<br/>Clifford scalar product]
        TOPK[Top-k Slot Selection]
        KVAPP[Append-Only K/V Cache]
        RopeDoc[Document-wise RoPE]
    end

    subgraph Output["Output Boundary"]
        LOGIT[Logits]
        LOSS[Composite Loss<br/>L_CE + L_aux + L_iso]
    end

    TOK --> EMB
    EMB --> TBLOCK
    TBLOCK --> HSTATE
    HSTATE --> LSTATE
    LSTATE --> TBLOCK
    LSTATE --> PROJ
    PROJ --> EXTR
    EXTR --> XFER
    XFER --> FUSE
    FUSE --> HSTATE
    EXTR --> RKEY
    RKEY --> TOPK
    TOPK --> KVAPP
    KVAPP --> RopeDoc
    RopeDoc --> TBLOCK
    TBLOCK --> LOGIT
    LOGIT --> LOSS
    MASK --> TBLOCK
    PART --> TBLOCK
```

**Contract boundaries**
- TensorRuntime boundary: every arrow carries `TensorSpec` (shape, dtype, layout, backend).
- HDIM bridge: `bridge_preserves_hidden_shape` guarantees fused output matches input hidden shape.
- MSA routing: `RouteWithinSlots` guarantees every selected `slotId` exists in registry.

---

## 2. HRM Recurrence Algorithm

```mermaid
flowchart LR
    subgraph Cycle["One H-Cycle"]
        direction TB
        zH_in["z_H (strategic)<br/>shape: [B, H_dim]"]
        zL_in["z_L (tactical)<br/>shape: [B, L_dim]"]

        subgraph L_Loop["L-Cycle Loop"]
            direction TB
            INP["Input tokens<br/>+ z_L concat"]
            ATT["Self-Attention<br/>PrefixLM mask"]
            MLP["Feedforward MLP"]
            zL_out["z_L <- output"]
            INP --> ATT --> MLP --> zL_out
        end

        zH_upd["z_H <- f(z_H, z_L_last)"]
        zL_reset["z_L <- g(z_H_new)"]

        zH_in --> zH_upd
        zL_in --> L_Loop
        L_Loop --> zH_upd
        zH_upd --> zL_reset
    end
```

**Algorithm (pseudocode)**

```
function HRMForward(tokens, z_H, z_L, prefix_mask, partition):
    for h in 1..H_cycles:
        for l in 1..L_cycles:
            x = Embed(tokens) + project_z_L(z_L)   // broadcast z_L to hidden dim
            x = TransformerBlock(x, mask=prefix_mask, partition=partition)
            z_L = UpdateL(x, z_L)          // tactical recurrence
        z_H = UpdateH(z_H, z_L)             // strategic update
        z_L = ResetL(z_H)                  // tactical reset from new strategy
    return logits, z_H, z_L
```

**Invariants**
- `shape(z_H)` constant across all `HTransition` calls.
- `shape(z_L)` constant across all `LTransition` calls.
- `PrefixLMLegal(q, k)` rejects prefix->suffix attention edges.
- `PackedPartition` guarantees no sequence overlap in batch.

---

## 3. HRM Training Pipeline

```mermaid
flowchart LR
    DATA["Tokenized sampled data<br/>corpus path equivalent to /dev/shm/sampled"]
    SAMPLER["Stratified sampler"]
    BATCH["Packed PrefixLM batch<br/>tokens + mask + partition"]
    HRM["HRM recurrence<br/>H_cycles x L_cycles"]
    HDIM["HDIM fusion<br/>project -> invariant -> transfer -> fuse"]
    LOSS["Composite loss<br/>L_CE + L_aux + L_iso"]
    OPT["Optimizer step"]
    CKPT["Checkpoint<br/>epoch dir + optimizer + config"]
    EVAL["Evaluation<br/>run_only=[MATH,DROP,ARC,MMLU]"]
    EXPORT["HF/Transformers export<br/>config-field mapping"]

    DATA --> SAMPLER --> BATCH --> HRM --> HDIM --> LOSS --> OPT
    OPT --> CKPT
    CKPT --> EVAL
    CKPT --> EXPORT
```

---

## 4. HDIM Clifford Pipeline

```mermaid
flowchart LR
    subgraph Projection["1. Projection"]
        H["Hidden tensor<br/>[B, T, hidden_size]"]
        W["Linear layer W_proj<br/>[hidden_size -> heads * bladeCount]"]
        G["Multivector G<br/>[B, T, heads, bladeCount]"]
        H --> W --> G
    end

    subgraph Extract["2. Invariant Extraction"]
        Rsrc["DomainRotor R_src<br/>+ inverse R_src^-1"]
        U["Invariant U<br/>U = R_src^-1 * G * R_src"]
        G --> Rsrc --> U
    end

    subgraph Transfer["3. Domain Transfer"]
        Rtgt["DomainRotor R_tgt<br/>+ inverse R_tgt^-1"]
        Gtgt["G_target<br/>G_target = R_tgt * U * R_tgt^-1"]
        U --> Rtgt --> Gtgt
    end

    subgraph Fuse["4. Gated Fusion"]
        GATE["Gate sigma(W_gate * [hidden || flatten(Gtgt)])"]
        RES["Residual: hidden + gate * W_fuse(flatten(Gtgt))"]
        H --> GATE --> RES
        Gtgt --> GATE
    end
```

**Algorithm (pseudocode)**

```
function HDIMForward(hidden, rotor_src, rotor_tgt):
    // 1. Project
    G = reshape(linear(hidden), [B, T, heads, bladeCount])

    // 2. Extract invariant (rotor sandwich)
    U = geometric_product(geometric_product(rotor_src.inverse, G), rotor_src.value)

    // 3. Transfer to target domain
    G_target = geometric_product(rotor_tgt.value,
                                 geometric_product(U, rotor_tgt.inverse))

    // 4. Gated fusion back to hidden
    gate = sigmoid(W_gate * concat(hidden, flatten(G_target)))
    fused = hidden + gate * W_fuse(flatten(G_target))
    return fused
```

**Proof obligation**
- `same_rotor_transfer_identity`: if `rotor_src == rotor_tgt` and rotor is `UnitRotor`, then `HDIMForward(hidden, r, r) == hidden` modulo `FloatApprox ε`.

---

## 5. MSA Sparse Routing Algorithm

```mermaid
flowchart TB
    subgraph Offline["Offline memory encoding"]
        DOC["Tokenized documents"]
        CHUNK["Chunk + pool"]
        CONTENT["Content cache<br/>(K̄, V̄)<br/>host memory"]
        RKEYBUILD["Routing-key cache<br/>K̄ᵣ"]
        SHARDS["Routing-key shards<br/>GPU resident"]
        DOC --> CHUNK
        CHUNK --> CONTENT
        CHUNK --> RKEYBUILD --> SHARDS
    end

    subgraph Online["Online routing"]
        Q["Active hidden"]
        Qproj["Project to Clifford"]
        Qinv["Extract invariant"]
        Qr["Routing query Qᵣ"]
        SCORE["Score Qᵣ against K̄ᵣ"]
        TOP["Top-k slot IDs"]
        Q --> Qproj --> Qinv --> Qr --> SCORE
        SHARDS --> SCORE --> TOP
    end

    subgraph FetchAttend["Async fetch + sparse attention"]
        FETCH["Async fetch selected K/V"]
        LOCAL["Local context K/V"]
        CONCAT["Concatenate sparse K/V + local context"]
        ATT["Sparse attention"]
        TOP --> FETCH
        CONTENT -.->|"host-to-GPU async fetch"| FETCH
        FETCH --> CONCAT
        LOCAL --> CONCAT --> ATT
    end
```

**Algorithm (pseudocode)**

```
function MSARoute(query_hidden, routing_key_shards, host_kv_cache, local_kv, k_select=5):
    Q_r = HDIM.ExtractInvariant(Project(query_hidden))
    scores = ScoreOnGpu(Q_r, routing_key_shards)
    selected = TopK(scores, k_select)

    assert all(s in host_kv_cache for s in selected)     // RouteWithinSlots
    assert len(set(selected)) == len(selected)           // No duplicates

    sparse_kv = AsyncFetch(host_kv_cache, selected)
    WaitForFetch(sparse_kv)
    K_cat, V_cat = concat(sparse_kv, local_kv, dim=seq)
    output = Attention(query_hidden, K_cat, V_cat)
    return output, selected
```

**Invariants**
- `CacheAppendOnly`: new slots append, old slots remain prefix.
- `SameRoPEDocument`: positions from different documents never mix.
- Routing-key shards are GPU-resident; content K/V pages remain in host memory until selected.

---

## 6. Memory Interleave Loop

```mermaid
flowchart LR
    START["Current generation state"]
    QUERY["Generate retrieval query"]
    ROUTE["Route top-k memory slots"]
    EXPAND["Expand context<br/>sparse memory + local context"]
    GEN["Generate next span"]
    STOP{"Stop condition met?"}
    END["Return output"]

    START --> QUERY --> ROUTE --> EXPAND --> GEN --> STOP
    STOP -->|"no"| QUERY
    STOP -->|"yes"| END

    MAX["Stop conditions:<br/>max rounds<br/>empty retrieval<br/>confidence threshold"]
    MAX -.-> STOP
```

---

## 7. End-to-End Data Flow with Memory Interleave

```mermaid
sequenceDiagram
    participant U as User Input
    participant HRM as HRM Backbone
    participant HDIM as HDIM Layer
    participant MSA as MSA Router
    participant MEM as Memory K/V Cache
    participant OUT as Output / Loss

    U->>HRM: Token IDs + PrefixLM mask
    HRM->>HRM: L-cycle loop (tactical)
    HRM->>HRM: H-cycle update (strategic)

    par HDIM Structural Path
        HRM->>HDIM: Hidden state [B,T,hidden]
        HDIM->>HDIM: Project -> Extract U
        HDIM->>HDIM: Transfer -> Fuse -> hidden
        HDIM-->>HRM: Fused hidden (residual)
    and MSA Memory Path
        HDIM->>MSA: Invariant U as routing key
        MSA->>MSA: Score against slot registry
        MSA->>MSA: Top-k selection + safety checks
        MSA->>MEM: Fetch K/V for selected slots
        MEM-->>MSA: K/V tensors
        MSA->>MSA: Document-wise RoPE + sparse attention
        MSA-->>HRM: Attention output
    end

    HRM->>OUT: Logits
    OUT->>OUT: L_CE (response-only cross-entropy)
    OUT->>OUT: L_aux (contrastive router alignment)
    OUT->>OUT: L_iso (transfer fidelity)
```

**Critical path**
- Training: HRM -> HDIM -> HRM -> Loss (no MSA needed per step, MSA is inference-optional for M4-M6).
- Inference with memory: HRM -> HDIM (parallel) -> MSA -> HRM -> logits.

---

## 8. HDIM Transfer State

```mermaid
flowchart LR
    H["hidden<br/>transient"]
    GS["g_source<br/>cached for loss/replay"]
    UI["u_inv<br/>cached invariant"]
    UM["u_mem<br/>memory transfer state"]
    UR["u_route<br/>routing state"]
    GT["g_target<br/>transient transfer output"]
    OUT["fused hidden<br/>output"]

    H --> GS --> UI
    UI --> UM --> GT
    UI --> UR --> GT
    GT --> OUT

    META["TransferState also stores:<br/>memory_loss, router_state, memory_mode"]
    META -.-> UM
    META -.-> UR
```

---

## 9. Composite Loss Computation

```mermaid
flowchart TB
    subgraph Inputs["Loss Inputs"]
        LOGIT["Logits [B, T, vocab]"]
        TARGET["Target tokens [B, T]"]
        PREFIX["Prefix mask [B, T]<br/>1 = prefix, 0 = response"]
        ROUTE["Routing weights [B, k]"]
        INV_SRC["Source invariant U_src"]
        INV_TGT["Target invariant U_tgt"]
    end

    subgraph L_CE["L_CE — Response-Only Cross-Entropy"]
        MASK_CE["Mask: prefix == 0"]
        CE["CrossEntropy(logits, targets)"]
        LOGIT --> CE
        TARGET --> CE
        PREFIX --> MASK_CE --> CE
    end

    subgraph L_AUX["L_aux — Supervised Contrastive"]
        POS["Positive pairs<br/>same-domain invariants"]
        NEG["Negative pairs<br/>different-domain invariants"]
        COS["Cosine similarity"]
        MARGIN["Margin loss"]
        ROUTE --> POS --> COS --> MARGIN
        ROUTE --> NEG --> COS --> MARGIN
    end

    subgraph L_ISO["L_iso — Isomorphic Fidelity"]
        DIFF["U_src - U_tgt"]
        SQ["Squared norm"]
        INV_SRC --> DIFF --> SQ
        INV_TGT --> DIFF --> SQ
    end

    subgraph NORM["MagicNorm-Clifford"]
        GRAD["Gradient tensor"]
        GRADE["Grade-wise norm<br/>per blade coefficient"]
        CLIP["Clip if > threshold"]
        GRAD --> GRADE --> CLIP
    end

    TOTAL["Total Loss = L_CE + lambda_aux * L_aux + lambda_iso * L_iso"]
    CE --> TOTAL
    MARGIN --> TOTAL
    SQ --> TOTAL
    CLIP -.->|"gradient clipping"| TOTAL
```

**Algorithm (pseudocode)**

```
function CompositeLoss(logits, targets, prefix_mask, router_weights,
                       U_src, U_tgt, lambda_aux, lambda_iso):
    // L_CE — only on response tokens
    response_positions = (prefix_mask == 0)
    L_CE = CrossEntropy(logits[response_positions],
                        targets[response_positions])

    // L_aux — router alignment: maximize similarity for matching invariants,
    // minimize for non-matching (different structure or domain)
    pos_sim = CosineSimilarity(U_src, U_matched)   // structurally matching pair
    neg_sim = CosineSimilarity(U_src, U_random)    // non-matching pair
    L_aux = Mean(ReLU(margin - pos_sim + neg_sim))

    // L_iso — transfer fidelity penalty
    L_iso = Mean((U_src - U_tgt)^2)

    // Total
    loss = L_CE + lambda_aux * L_aux + lambda_iso * L_iso

    // Gradient norm bound (MagicNorm-Clifford)
    grads = Autograd(loss)
    for grad in grads:
        if CliffordGradeNorm(grad) > threshold:
            grad *= threshold / CliffordGradeNorm(grad)

    return loss
```

**Contracts**
- `L_CE` finite and non-NaN on toy batch.
- `L_aux` increases cosine of matching invariants, decreases for negatives.
- `L_iso = ||U_src - U_tgt||^2` penalizes transfer discrepancy.
- Gradient norms bounded by `MagicNorm-Clifford` to prevent blade-coefficient explosion.

---

## 10. Milestone Dependency Graph

```mermaid
flowchart TD
    M0["M0: Tensor Runtime & Core Types<br/>Shape, Layout, DType, CPU Tensor"]
    M1["M1: Clifford Algebra Core<br/>Multivector, Rotor, Product Table"]
    M2["M2: HRM Backbone<br/>Transformer, PrefixLM, RoPE"]
    M3["M3: HDIM Layer<br/>Projection, Invariant, Transfer, Fusion"]
    M4["M4: MSA Sparse Memory<br/>Routing, Top-k, RoPE, K/V Cache"]
    M5["M5: Composite Loss & Training<br/>L_CE + L_aux + L_iso, MagicNorm"]
    M6["M6: CUDA-Oxide Kernels<br/>Geometric product, rotor sandwich, sparse attn"]

    M0 --> M1
    M0 --> M2
    M1 --> M3
    M2 --> M3
    M2 --> M4
    M3 --> M5
    M4 --> M5
    M1 --> M6
    M4 --> M6

```

**Critical path (training)**: M0 -> M1 -> M3 -> M5
**Secondary path (inference scaling)**: M0 -> M2 -> M4 -> M6

---

## 11. Verification Stack

```mermaid
flowchart LR
    subgraph Lean["Lean4 Verification Layer"]
        CT["CoreTypes.lean<br/>Shape safety, index bijection"]
        HD["HDIM.lean<br/>Unit rotor, transfer identity"]
        HR["HRM.lean<br/>Shape preservation, PrefixLM legality"]
        MS["MSA.lean<br/>Append-only, route safety"]
    end

    subgraph Rust["Rust Implementation"]
        RTC["core-types + tensor-runtime"]
        RC["clifford-core"]
        RH["hrm-model"]
        RD["hdim-model"]
        RM["msa"]
        RL["losses"]
    end

    subgraph Tests["Test & Property Verification"]
        T0["Shape round-trip"]
        T1["Cayley table correctness"]
        T2["Rotor sandwich ≈ identity"]
        T3["PrefixLM mask unit tests"]
        T4["Fusion shape check"]
        T5["Top-k slots exist"]
        T6["RoPE separation check"]
        T7["Loss finite, grad non-NaN"]
        T8["CPU vs CUDA ≈ ε"]
    end

    CT --> RTC --> T0
    HD --> RC --> T1
    HD --> RD --> T2
    HR --> RH --> T3
    HD --> RD --> T4
    MS --> RM --> T5
    MS --> RM --> T6
    RL --> T7
    RM --> T8
```

---

## 12. Formal Verification Traceability

```mermaid
flowchart LR
    subgraph Lean["Lean theorem / contract"]
        L1["unit_rotor_sandwich_identity"]
        L2["norm preservation"]
        L3["HRM recurrence depth monotonicity"]
        L4["RouteWithinSlots"]
        L5["Tensor shape/dtype/layout preservation"]
    end

    subgraph Rust["Rust crate node"]
        R1["clifford-core"]
        R2["hdim-model"]
        R3["hrm-model"]
        R4["msa"]
        R5["tensor-runtime"]
    end

    subgraph Tests["Parity / golden tests"]
        T1["rotor_sandwich_identity golden"]
        T2["norm_preservation golden"]
        T3["recurrent_depth_monotonicity test"]
        T4["route_within_slots test"]
        T5["shape_dtype_layout golden"]
    end

    GATE["Backend dispatch gate<br/>CPU/CUDA parity required"]

    L1 --> R1 --> T1 --> GATE
    L1 --> R2 --> T1
    L2 --> R1 --> T2 --> GATE
    L3 --> R3 --> T3 --> GATE
    L4 --> R4 --> T4 --> GATE
    L5 --> R5 --> T5 --> GATE
```

For milestone dependencies, risk analysis, and stop-conditions see [`implementation_plan.md`](implementation_plan.md).
For algorithmic complexity, resource estimates, and realizability verification see [`realizability_verification.md`](realizability_verification.md).
