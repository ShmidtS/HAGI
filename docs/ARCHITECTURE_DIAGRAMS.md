# HAGI Architecture Diagrams

Mermaid-compatible diagrams mapping to HAGI subsystems and pipeline stages.

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph Input["Input Boundary"]
        TOK[Token IDs + Position IDs]
        MASK[PrefixLM Mask]
    end

    subgraph Perception["PERCEPTION (2 layers, unique)"]
        EMB[Token Embedding 49K -> 576]
        PBLOCK[TransformerBlock<br/>RMSNorm -> GQA -> RMSNorm -> MoE]
    end

    subgraph Reasoning["REASONING CORE (7 layers, HRM-controlled)"]
        direction TB
        HRM["HRM Loop<br/>H_cycles x L_cycles"]
        HDIM["HDIM<br/>project -> invariant -> transfer -> fuse"]
        GDR["GDR<br/>decompose -> grade update -> geometric product -> recompose"]
        TBLOCK["TransformerBlock<br/>RMSNorm -> GQA -> RMSNorm -> MoE"]
        MSA["MSA<br/>slot routing + sparse attention"]
        HSTATE["H/L State Transitions<br/>z_H (strategic) / z_L (tactical)"]
    end

    subgraph Expression["EXPRESSION (2 layers, unique)"]
        EBLOCK[TransformerBlock]
    end

    subgraph Output["Output Boundary"]
        NORM[RMSNorm]
        LMHEAD["LM Head 576 -> 49K (tied)<br/>or CAST: K=8 block prediction"]
        LOSS["Composite Loss<br/>L_CE + L_iso + L_moe + L_msa_lb + L_gdr"]
    end

    TOK --> EMB --> PBLOCK
    PBLOCK --> HRM
    HRM --> HDIM --> GDR --> TBLOCK --> MSA --> HSTATE
    HSTATE --> HRM
    HRM --> EBLOCK
    EBLOCK --> NORM --> LMHEAD --> LOSS
    MASK --> TBLOCK
```

---

## 2. HRM Recurrence Algorithm

```mermaid
flowchart LR
    subgraph Cycle["One H-Cycle"]
        direction TB
        zH_in["z_H (strategic)<br/>shape: [B, h_dim]"]
        zL_in["z_L (tactical)<br/>shape: [B, l_dim]"]

        subgraph L_Loop["L-Cycle Loop (default 2)"]
            direction TB
            INP["Input tokens + z_L concat"]
            HDIM_STEP["HDIM: project -> invariant -> transfer"]
            GDR_STEP["GDR: decompose -> grade update -> geo product"]
            ATT["Self-Attention (GQA + fp16)"]
            MLP["MoE SwiGLU + MoD skip"]
            MSA_STEP["MSA: slot routing + sparse attention<br/>(memory-aware: read+write per cycle)"]
            zL_out["z_L <- UpdateL(output)"]
            INP --> HDIM_STEP --> GDR_STEP --> ATT --> MLP --> MSA_STEP --> zL_out
        end

        zH_upd["z_H <- UpdateH(z_H, z_L_last)"]
        zL_reset["z_L <- ResetL(z_H_new)"]

        zH_in --> zH_upd
        zL_in --> L_Loop
        L_Loop --> zH_upd
        zH_upd --> zL_reset
    end
```

**Algorithm (pseudocode)**

```
function HRMForward(tokens, z_H, z_L, prefix_mask):
    for h in 1..H_cycles (default 1):
        for l in 1..L_cycles (default 2):
            x = Embed(tokens) + project_z_L(z_L)
            x = HDIM(x)           # invariant transfer
            x = GDR(x)            # grade-decomposed update
            x = TransformerBlock(x, mask=prefix_mask)
            x = MSA(x)            # memory-aware slot routing
            z_L = UpdateL(x, z_L) # tactical recurrence
        z_H = UpdateH(z_H, z_L)   # strategic update
        z_L = ResetL(z_H)         # tactical reset
    return logits, z_H, z_L
```

---

## 3. GDR Grade Decomposition

```mermaid
flowchart LR
    subgraph Decompose["1. Decomposition"]
        H["Hidden [B, T, 576]"]
        S["Scalar 64<br/>(confidence)"]
        V["Vector 96<br/>(entities)"]
        B["Bivector 96<br/>(relations)"]
        T["Trivector 64<br/>(structure)"]
        R["Residual 256<br/>(unconstrained)"]
        H --> S & V & B & T & R
    end

    subgraph Update["2. Per-Grade Update"]
        CTX["Shared context<br/>concat(S, V, B, T)"]
        SU["0.8*S + 0.2*MLP_S<br/>(slow momentum)"]
        VU["0.5*V + 0.5*MLP_V<br/>(medium momentum)"]
        BU["MLP_B<br/>(full update)"]
        TU["MLP_T<br/>(full update)"]
        CTX --> SU & VU & BU & TU
    end

    subgraph Geo["3. Geometric Interaction"]
        GP["geometric_product_self_g02<br/>(Cl(3,0,0), 8 blades)"]
        G0["gate_0 * grade_0<br/>(scalar output)"]
        G2["gate_2 * grade_2<br/>(bivector output)"]
        GP --> G0 & G2
    end

    subgraph Recompose["4. Recomposition"]
        NEW["Hidden [B, T, 576]<br/>= concat(S', V', B', T', R)"]
    end

    S & V & B & T --> CTX
    SU --> GP
    VU --> GP
    G0 --> NEW
    G2 --> NEW
    SU --> NEW
    VU --> NEW
    BU --> NEW
    TU --> NEW
    R --> NEW
```

---

## 4. HDIM Domain Transfer Pipeline

```mermaid
flowchart LR
    subgraph Projection["1. Projection"]
        H["Hidden [B, T, 576]"]
        W["HiddenToMultivector<br/>Linear(576 -> heads*8)"]
        G["Multivector G<br/>[B, T, 4, 8]"]
        H --> W --> G
    end

    subgraph Extract["2. Invariant Extraction"]
        Rsrc["DomainRotor R_src"]
        U["Invariant U<br/>U = R_src^-1 * G * R_src"]
        G --> Rsrc --> U
    end

    subgraph Transfer["3. Domain Transfer"]
        Rtgt["DomainRotor R_tgt"]
        Gtgt["G_target<br/>G_target = R_tgt * U * R_tgt^-1"]
        U --> Rtgt --> Gtgt
    end

    subgraph Fuse["4. Gated Fusion"]
        GATE["Gate = sigmoid(W_gate * [hidden || flatten(Gtgt)])"]
        RES["Residual: hidden + gate * W_fuse(flatten(Gtgt))"]
        H --> GATE --> RES
        Gtgt --> GATE
    end
```

**Rotor Schedule**: 4 parallel rotors, LCG-based index selection (deterministic, no GPU sync). `rotor_seed=42` for reproducibility.

---

## 5. MSA Sparse Routing

```mermaid
flowchart TB
    subgraph Registry["Slot Registry (persistent at inference)"]
        SLOTS["Memory Slots<br/>slot_id, domain_id, routing_key, K/V cache"]
        EVICT["Eviction (oldest first)<br/>when slot_count exceeded"]
    end

    subgraph Online["Online Routing"]
        Q["Active hidden state"]
        QPROJ["HDIM invariant extraction<br/>(routing query Q_r)"]
        SCORE["Score: Q_r . K_bar_r<br/>(dot product)"]
        TOPK["Top-k selection (k=6)"]
        Q --> QPROJ --> SCORE
        SLOTS --> SCORE
        SCORE --> TOPK
    end

    subgraph Attend["Sparse Attention"]
        FETCH["Fetch selected K/V pages"]
        LOCAL["Local context K/V"]
        CONCAT["Concat: sparse K/V + local K/V"]
        ATT["Sparse attention"]
        TOPK --> FETCH
        SLOTS -.->|"async fetch"| FETCH
        FETCH --> CONCAT
        LOCAL --> CONCAT --> ATT
    end
```

**Training**: registry cleared every forward pass.
**Inference**: persistent registry accumulates across decode steps.

---

## 6. MoE + Mixture-of-Depths

```mermaid
flowchart LR
    IN["Hidden [B, T, 576]"]
    ROUTER["Router<br/>Linear(576 -> num_experts + 1)"]
    SOFTMAX["Softmax / Top-k (k=1)"]
    SKIP{"Skip slot wins?"}
    EXPERTS["Expert 0..3<br/>SwiGLU 576 -> 384 -> 576"]
    SKIP_OUT["Identity (output 0)"]
    COMBINE["Weighted sum"]
    OUT["Hidden [B, T, 576]"]

    IN --> ROUTER --> SOFTMAX
    SOFTMAX --> SKIP
    SKIP -->|"yes"| SKIP_OUT --> OUT
    SKIP -->|"no"| EXPERTS --> COMBINE --> OUT
    IN --> EXPERTS
```

**MoD skip**: trivial tokens bypass experts entirely (residual identity). Skip slot excluded from load-balance aux loss.

---

## 7. CAST Block Generation

```mermaid
flowchart LR
    H["Hidden [B, T, 576]"]
    PROJ["block_proj<br/>Linear(576 -> K*576)"]
    RESHAPED["[B, T, K=8, 576]"]
    MV["Multivectors<br/>[B, T, K, 72, 8]"]
    GEO["geometric_product<br/>(adjacent K positions)"]
    AREA["Bivector area<br/>(cross-token coherence)"]
    MOD["Area modulates<br/>neighbour virtual states"]
    FLAT["[B, T, K, 576]"]
    DECODE["final_norm + lm_head<br/>(shared, per k position)"]
    TOKENS["K=8 token predictions"]

    H --> PROJ --> RESHAPED --> MV --> GEO --> AREA --> MOD --> FLAT --> DECODE --> TOKENS
```

**Training**: `train_k=3` subsamples CE loss positions (always k=0 + 2 random from k=1..7).
**Inference**: all K=8 positions used. 8x fewer forward passes.

---

## 8. Composite Loss

```mermaid
flowchart TB
    subgraph Inputs["Loss Inputs"]
        LOGIT["Fused CE: hidden -> loss<br/>(no logits materialization)"]
        TARGET["Target tokens"]
        INV["HDIM invariants<br/>U_src, U_tgt"]
        ROUTE["MoE routing weights"]
        MSAR["MSA routing weights"]
        GDRR["GDR router weights"]
    end

    subgraph L_CE["L_CE (w=1.0)"]
        FCE["fused_linear_cross_entropy<br/>label_smoothing=0.05"]
    end

    subgraph L_ISO["L_iso (w=0.02)"]
        ISO["MSE(U_src, U_tgt)"]
    end

    subgraph L_MOE["L_moe (w=0.005)"]
        MOELB["Shazeer/Switch load-balance"]
    end

    subgraph L_MSA["L_msa_lb (w=0.01)"]
        MSALB["MSA router load-balance"]
    end

    subgraph L_GDR["L_gdr_router (w=0.005)"]
        GDRLB["GDR capacity router load-balance"]
    end

    TOTAL["L_total = L_CE + 0.02*L_iso + 0.005*L_moe<br/>+ 0.01*L_msa_lb + 0.005*L_gdr_router"]
    FCE --> TOTAL
    ISO --> TOTAL
    MOELB --> TOTAL
    MSALB --> TOTAL
    GDRLB --> TOTAL
```

**Warmup**: all auxiliary losses ramp from 0 to target over 1000 steps.

---

## 9. Training Pipeline

```mermaid
flowchart LR
    DATA["Sequential Cycling<br/>easy -> mid -> hard curriculum"]
    BATCH["PrefixLM Batch<br/>tokens + mask + partition"]
    TEACHER["SmolLM2-135M Teacher<br/>(freed at 60% training)"]
    STUDENT["HAGI Forward<br/>HRM x HDIM x GDR x MSA x MoE"]
    DISTILL["KL Distillation<br/>alpha*CE + (1-alpha)*T^2*KL"]
    LOSS["Composite Loss"]
    OPT["Muon (2D) + AdamW (1D)<br/>WSD schedule"]
    CKPT["Checkpoint<br/>step-NNNNNN.pt"]
    EVAL["Evaluation<br/>lm-eval-harness"]

    DATA --> BATCH --> STUDENT
    TEACHER -.-> DISTILL
    STUDENT --> LOSS
    DISTILL -.-> LOSS
    LOSS --> OPT --> CKPT
    CKPT --> EVAL
```

---

## 10. Reasoning Cache Decoding

```mermaid
flowchart LR
    PROMPT["User prompt"]
    TURN1["Turn 1"]
    GEN1["Generate reasoning trace<br/>z_R (H_R=512 tokens)"]
    SUM1["Summarize<br/>z_S (H_S=128 tokens)"]
    CACHE1["Cache z_S^(1)"]

    TURN2["Turn 2"]
    GEN2["Generate reasoning trace<br/>conditioned on prompt + z_S^(1)"]
    SUM2["Summarize z_S^(2)"]
    CACHE2["Cache z_S^(2)"]

    TURN3["Turn 3 (final)"]
    GEN3["Generate final response"]
    OUTPUT["Output"]

    PROMPT --> TURN1
    TURN1 --> GEN1 --> SUM1 --> CACHE1
    CACHE1 --> TURN2
    TURN2 --> GEN2 --> SUM2 --> CACHE2
    CACHE2 --> TURN3
    TURN3 --> GEN3 --> OUTPUT
```

**MSA integration**: summary hidden states registered as MSA slots for cross-iteration memory retrieval.

---

## 11. Data Pipeline

```mermaid
flowchart LR
    subgraph Sources["Data Sources"]
        TS["TinyStories"]
        PI["Python Instruct"]
        ST["SmolTalk"]
        WE["Wikipedia EN"]
        WR["Wikipedia RU"]
        OM["OpenWebMath"]
        OR["OSCAR RU"]
        SP["SlimPajama"]
        EDU["FineWeb EDU"]
    end

    subgraph Tokenize["Tokenization"]
        TK["SmolLM2 tokenizer<br/>(49,152 vocab)"]
        BIN[".bin memmap files<br/>(uint16, BFD packing)"]
    end

    subgraph Curriculum["Sequential Cycling (3x)"]
        P1["Phase 1: Easy<br/>TS + PI + ST"]
        P2["Phase 2: Mid<br/>WE + WR + OM"]
        P3["Phase 3: Hard<br/>OR + SP + EDU"]
    end

    subgraph Loader["Data Loading"]
        MM["MemmapDataset"]
        SC["SequentialCyclingIterator"]
        BATCH["PrefixLM Batch<br/>B=10, T=1024"]
    end

    TS & PI & ST & WE & WR & OM & OR & SP & EDU --> TK --> BIN
    BIN --> P1 --> P2 --> P3
    P1 & P2 & P3 --> MM --> SC --> BATCH
```
