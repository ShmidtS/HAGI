# HAGI Milestones

Staged roadmap. Each stage adds **one** variable. No stage proceeds until its gate passes. Every stage has an explicit stop/pivot condition — research that cannot fail is not research.

The guiding rule: **validate the hypothesis cheaply before investing in infrastructure.** PyTorch prototype reaches a result in ~6-8 weeks. Rust/CUDA/formal-verification work happens only after the architecture is proven.

---

## Stage 0 — Dense Baseline

**Goal:** A working ~115M-parameter dense transformer, trained and benchmarked. This is the control against which everything is measured.

**Build:**
- 12-layer dense transformer (768 hidden, 12/4 GQA, SwiGLU, RoPE, RMSNorm)
- Data pipeline: 5-10B tokens (FineWeb-Edu + code + math)
- Training loop with logging, checkpointing, eval hooks
- Benchmark harness: GSM8K, ARC-Challenge, BoolQ, HellaSwag, WinoGrande, HumanEval, MMLU, perplexity

**Measure:** Held-out perplexity, all benchmark scores, training stability.

**Gate:**
- Training stable (no NaN, no divergence)
- Perplexity decreases to reasonable level (<20 on validation)
- Benchmarks produce non-random scores

**Stop/pivot if:** Cannot train stably → debug data pipeline and config. This is infrastructure, not research. Do not proceed to Stage 1 with a broken baseline.

**Estimated effort:** 2-3 weeks.

---

## Stage 1 — Recurrent Reasoning Core

**Goal:** Replace layers 5-8 with a parameter-shared block looped 3×. Flat hidden state (no Clifford yet). This isolates the benefit of recurrent depth.

**Build:**
- Looped block over layers 5-8, shared parameters
- Loop iteration embedding
- Truncated BPTT through the loop
- (Optional 1b) PonderNet-style learned halting

**Measure:** GSM8K delta vs Stage 0 (primary), ARC-C delta, perplexity delta. Probe: what differs between iteration 1 and iteration 3 hidden states?

**Gate:**
- Training stable with loops
- GSM8K improves ≥3% absolute over Stage 0
- Perplexity does not regress >3%

**Stop/pivot if:**
- Zero GSM8K improvement → try looping different layers (1-4 vs 9-12), try 2×/5× loops
- Perplexity degrades >5% → reduce LR for looped params, detach gradients earlier
- Training diverges → gradient clipping per-loop, stop-gradient between iterations

**Estimated effort:** 2-3 weeks.

---

## Stage 2 — Grade-Decomposed Recurrence (GDR)

**Goal:** The core HAGI contribution. Decompose the hidden state into Clifford grades within the reasoning loop, with per-grade update dynamics and geometric-product cross-grade interaction.

**Build:**
- `Cl(3,0,0)` geometric product (8×8 Cayley table)
- Grade decomposition: 768 → [64 scalar, 192 vector, 192 bivector, 64 trivector, 256 residual]
- Per-grade update MLPs with momentum (scalar α=0.9, vector α=0.5, bivector/trivector full update)
- Geometric interaction layer
- Recomposition

**Train all four models for the ablation:**

| Model | Architecture |
|-------|-------------|
| A | Dense baseline (from Stage 0) |
| B | Looped, flat (from Stage 1) |
| C | Dense + Clifford projection bolted on |
| D | Looped + grade-decomposed recurrence (full GDR) |

**Measure:** Full ablation matrix. Focus on relational/directional tasks (ARC-C, BoolQ, WinoGrande). Examine gate activations — is the model using or ignoring the Clifford signal?

**Critical comparisons:**
- **B vs D** — does grade decomposition add value to recurrence? (the key result)
- **C vs D** — does integrating Clifford into recurrence beat bolting it on?

**Gate (success):**
- Model D outperforms both B and C on ≥2 reasoning benchmarks by ≥2% absolute
- Model D perplexity within 3% of Model A

**Stop/pivot if:**
- D ≈ B → grade decomposition neutral. Fall back to publishing B + training recipe.
- D < B → grade decomposition harmful. Investigate momentum coefficients, grade partition, residual size.
- Gate values ≈ 0 → model ignores Clifford. Projection may lose info. Try `Cl(4,0,0)`, remove gate, or cross-attention fusion.

**Estimated effort:** 3-4 weeks.

> **This is the make-or-break stage.** A positive result here is the publishable contribution. Everything after is engineering and scaling.

---

## Stage 3 — Context Efficiency

**Goal:** Add MLA-style KV cache compression and extend context. Deployment efficiency, not capability.

**Build:**
- Multi-head Latent Attention (latent KV dim ~128, expanded to 768)
- Extended context to 8K-16K
- (Optional) lightweight persistent memory slots

**Measure:** Benchmark parity with Stage 2 (MLA must not degrade quality), KV cache memory reduction, inference throughput.

**Gate:**
- Quality parity within 2% of Stage 2 model
- KV cache reduced ≥6× (combined with GQA's 3×)

**Stop/pivot if:** MLA degrades quality >2% → increase latent dim to 192/256. Memory slots don't help → remove them.

**Estimated effort:** 4 weeks.

---

## Stage 4 — Quantization Path

**Goal:** Quantize for local deployment. Measure quality retention.

**Build:**
- Q4_K_M and Q2_K GGUF export
- llama.cpp inference compatibility
- (Optional) quantization-aware fine-tuning for final epochs

**Measure:** Quality retention ratio (4-bit score / bf16 score), inference speed on target hardware, memory footprint.

**Gate:**
- 4-bit retains ≥95% of bf16 benchmark scores
- Model runs on consumer GPU (8-16GB) / CPU

**Stop/pivot if:** 4-bit drops >5% → use Q5_K/Q6_K. Reasoning degrades more than knowledge → mixed-precision (keep reasoning layers higher precision).

**Estimated effort:** 3 weeks.

---

## Stage 5 — Rust / CUDA Port

**Goal:** Production implementation of the validated architecture. Only after Stage 2 confirms the hypothesis.

**Build:**
- Port validated architecture to Rust (existing `crates/` workspace)
- Clifford geometric product CUDA kernel (`cudarc` or cuda-oxide)
- Optimized looped inference path
- CPU/CUDA golden-output parity

**Measure:** Throughput vs llama.cpp, memory usage, correctness parity with PyTorch reference.

**Gate:**
- Output matches PyTorch reference within tolerance
- Performance competitive with llama.cpp

**Stop/pivot if:** cuda-oxide fails to compile → use `cudarc` or `wgpu`. Rust 2× slower → correctness first, optimize later.

**Estimated effort:** 6-8 weeks.

---

## Stage 6 — Formal Verification Alignment

**Goal:** Connect existing Lean4 proofs to the Rust implementation. Make the proofs load-bearing.

**Build:**
- Property-based tests (`proptest`) mirroring each Lean theorem
- Verify Clifford product table, rotor sandwich identity, shape preservation through loops
- CI gate enforcing Lean ↔ Rust agreement

**Measure:** Every Lean theorem has a passing Rust property test.

**Gate:** All property tests pass. Runtime Clifford ops never violate a Lean invariant.

**Stop/pivot if:** No failure mode — pure engineering. Do when time allows.

**Estimated effort:** 2-4 weeks.

---

## Critical Path

```
Stage 0 ──► Stage 1 ──► Stage 2 ──► [DECISION POINT]
                                          │
                          ┌───────────────┴───────────────┐
                     (positive)                       (negative)
                          │                               │
                          ▼                               ▼
              Stage 3 ──► Stage 4 ──► Stage 5 ──► 6   Redesign GDR
                                                       (new hypothesis)
```

**First research result: Stage 2, ~6-8 weeks from start.**
**Working Rust prototype (if validated): ~20 weeks.**

## Mapping to GitHub Milestones

When pushed, create GitHub milestones matching these stages:

| GitHub Milestone | Stage | Due |
|------------------|-------|-----|
| `M0: Dense Baseline` | 0 | +3 weeks |
| `M1: Recurrent Core` | 1 | +6 weeks |
| `M2: Grade-Decomposed Recurrence` | 2 | +10 weeks |
| `M3: Context Efficiency` | 3 | +14 weeks |
| `M4: Quantization` | 4 | +17 weeks |
| `M5: Rust/CUDA Port` | 5 | +25 weeks |
| `M6: Formal Verification` | 6 | +29 weeks |
