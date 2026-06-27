# HAGI Milestones

Staged roadmap. Each stage adds one variable. No stage proceeds until its gate passes. Every stage has an explicit stop/pivot condition -- research that cannot fail is not research.

The guiding rule: **validate the hypothesis cheaply before investing in infrastructure.**

---

## Stage 0 -- Dense Baseline

**Goal:** A working ~74M-parameter dense transformer, trained and benchmarked. This is the control against which everything is measured.

**Build:**
- 2+7+2 layer dense transformer (576 hidden, 8/4 GQA, SwiGLU/MoE, RoPE, RMSNorm)
- Data pipeline: 3B tokens (FineWeb-Edu + code + math + Russian + instructions)
- Training loop with Muon+AdamW, distillation, sequential cycling, checkpointing
- Benchmark harness: GSM8K, ARC-Challenge, BoolQ, HellaSwag, WinoGrande

**Measure:** Held-out perplexity, all benchmark scores, training stability.

**Gate:**
- Training stable (no NaN, no divergence)
- Perplexity decreases to reasonable level
- Benchmarks produce non-random scores
- VRAM fits in 8GB with distillation teacher

**Status:** Implementation complete. Training in progress.

**Estimated effort:** 2-3 weeks.

---

## Stage 1 -- Recurrent Reasoning Core

**Goal:** HRM two-level reasoning loop (H-cycles x L-cycles). Flat hidden state (no Clifford yet). This isolates the benefit of recurrent depth.

**Build:**
- HRM with z_H (strategic) and z_L (tactical) states
- Loop iteration embedding
- Truncated BPTT through the loop
- Memory-aware HRM (MSA read+write inside L-cycle)
- Stochastic depth regularization

**Measure:** GSM8K delta vs Stage 0 (primary), ARC-C delta, perplexity delta. Probe: what differs between iteration 1 and iteration 2 hidden states?

**Gate:**
- Training stable with HRM loops
- GSM8K improves >=3% absolute over Stage 0
- Perplexity does not regress >3%

**Stop/pivot if:**
- Zero GSM8K improvement -> try different layer splits, try 1/3 L-cycles
- Perplexity degrades >5% -> reduce LR for looped params, detach gradients earlier
- Training diverges -> gradient clipping per-loop, stop-gradient between iterations

**Estimated effort:** 2-3 weeks.

---

## Stage 2 -- Grade-Decomposed Recurrence (GDR)

**Goal:** The core HAGI contribution. Decompose the hidden state into Clifford grades within the reasoning loop, with per-grade update dynamics and geometric-product cross-grade interaction.

**Build:**
- `Cl(3,0,0)` geometric product (8-blade Cayley table)
- Grade decomposition: 576 -> [64 scalar, 96 vector, 96 bivector, 64 trivector, 256 residual]
- Per-grade update MLPs with momentum (scalar 0.8, vector 0.5, bivector/trivector full update)
- Geometric interaction layer
- Learnable GDR capacity router
- Recomposition

**Train all four models for the ablation:**

| Model | Architecture |
|-------|-------------|
| A | Dense baseline (from Stage 0) |
| B | Looped, flat (from Stage 1) |
| C | Dense + Clifford projection bolted on |
| D | Looped + grade-decomposed recurrence (full HAGI) |

**Measure:** Full ablation matrix. Focus on relational/directional tasks (ARC-C, BoolQ, WinoGrande). Examine gate activations -- is the model using or ignoring the Clifford signal?

**Critical comparisons:**
- **B vs D** -- does grade decomposition add value to recurrence? (the key result)
- **C vs D** -- does integrating Clifford into recurrence beat bolting it on?

**Gate (success):**
- Model D outperforms both B and C on >=2 reasoning benchmarks by >=2% absolute
- Model D perplexity within 3% of Model A

**Stop/pivot if:**
- D ~= B -> grade decomposition neutral. Fall back to publishing B + training recipe.
- D < B -> grade decomposition harmful. Investigate momentum coefficients, grade partition, residual size.
- Gate values ~= 0 -> model ignores Clifford. Projection may lose info. Try `Cl(4,0,0)`, remove gate, or cross-attention fusion.

**Estimated effort:** 3-4 weeks.

> **This is the make-or-break stage.** A positive result here is the publishable contribution.

---

## Stage 3 -- HDIM Domain Transfer

**Goal:** Validate that HDIM cross-domain invariant transfer improves reasoning when integrated with GDR.

**Build:**
- Full HDIM pipeline: project -> invariant extraction -> domain transfer -> gated fusion
- Domain rotors (4 parallel rotors, LCG-scheduled)
- Delayed HDIM aggregation
- L_iso loss (invariant alignment)

**Measure:** Delta vs Stage 2 (D model without HDIM cross-domain). Examine rotor utilization -- are different rotors being used for different reasoning contexts?

**Gate:**
- HDIM improves >=1 reasoning benchmark by >=1% absolute
- L_iso decreases over training (invariant alignment is being learned)

**Stop/pivot if:**
- No improvement -> HDIM is neutral. Keep it disabled, publish GDR-only results.
- L_iso diverges -> rotor initialization or transfer formula incorrect.

**Estimated effort:** 2-3 weeks.

---

## Stage 4 -- MSA + MoE + CAST Integration

**Goal:** Validate the full architecture stack: GDR + HRM + HDIM + MSA + MoE + CAST.

**Build:**
- MSA with persistent slot registry at inference
- MoE with mixture-of-depths skip
- CAST block generation (K=8)
- Adaptive MSA top_k (MoD-guided)
- Reasoning Cache integration

**Measure:** Full architecture vs Stage 3. Inference speed (CAST 8x reduction). Memory context (MSA 16K effective tokens).

**Gate:**
- No quality regression from Stage 3
- CAST produces coherent multi-token output
- MSA persistent registry improves multi-turn conversations

**Stop/pivot if:**
- MoE routing collapses -> increase alpha, try top-2
- CAST coherence is poor -> disable coherence gate, reduce K
- MSA routing is unused -> disable adaptive top_k, increase top_k

**Estimated effort:** 2-3 weeks.

---

## Stage 5 -- RL Fine-Tuning

**Goal:** MGPO RL training to improve reasoning on math/code benchmarks.

**Build:**
- MGPO loop with group-relative advantage
- Prompt difficulty weighting
- Long2Short reward shift
- Reward shaping for reasoning quality

**Measure:** GSM8K delta vs Stage 4 (primary), ARC-C delta, HumanEval delta.

**Gate:**
- GSM8K improves >=5% absolute
- No reward hacking (response length, repetition)
- Training stable on 8GB

**Stop/pivot if:**
- No improvement -> RL adds no value at this scale. Publish SFT results.
- Reward hacking -> add length penalty, repetition detection.
- OOM -> reduce group size, reduce max_new_tokens.

**Estimated effort:** 3-4 weeks.

---

## Stage 6 -- Scaling and Deployment

**Goal:** Scale beyond RTX 3070 and prepare for deployment.

**Build:**
- Scale to 12GB / 16GB / 24GB configs (see VRAM scaling table in README)
- Quantization (Q4_K_M GGUF, llama.cpp compatibility)
- Extended context (8K-16K) via MLA
- Production inference optimization

**Measure:** Quality retention at scale, quantization quality retention, inference throughput.

**Gate:**
- 12GB config matches or beats 8GB config quality
- 4-bit retains >=95% of bf16 benchmark scores
- Model runs on consumer GPU (8-16GB) / CPU

**Estimated effort:** 4-6 weeks.

---

## Critical Path

```
Stage 0 --> Stage 1 --> Stage 2 --> [DECISION POINT]
                                         |
                         +---------------+---------------+
                    (positive)                      (negative)
                         |                               |
                         v                               v
             Stage 3 --> Stage 4 --> Stage 5 --> 6   Redesign GDR
                                                    (new hypothesis)
```

**First research result: Stage 2, ~6-8 weeks from start.**
**Full architecture validation: Stage 4, ~14 weeks.**
**RL + deployment: Stage 6, ~24 weeks.**

---

## Current Status

| Stage | Status |
|-------|--------|
| 0 | Implementation complete, training in progress |
| 1 | Implemented (HRM with memory-aware, stochastic depth) |
| 2 | Implemented (GDR with capacity router, geometric product) |
| 3 | Implemented (HDIM with rotors, delayed aggregation) |
| 4 | Implemented (MSA, MoE+MoD, CAST, RC) |
| 5 | Implemented (MGPO loop) |
| 6 | Not started |

All mechanisms are implemented and configurable. The ablation (Models A/B/C/D) can be run by toggling `use_loop` and `use_gdr` in the config.
