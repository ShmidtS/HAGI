# HAGI Architecture Realizability Verification

This document verifies that the HAGI architecture and its milestone decomposition from `implementation_plan.md` are technically realizable. Analysis covers algorithmic complexity, critical path duration, risk surface, and resource estimates.

---

## 1. Feasibility Verdict Summary

Architecture is **realizable** under milestone constraints defined in `implementation_plan.md`.  
Critical path **M0 → M1 → M3 → M5** is low-to-medium risk; high-risk items (M4 at scale, M6) are isolated on the inference-scaling secondary path and do not block training loop completion (M5).

Per-milestone risk levels and stop-conditions are listed in `implementation_plan.md`.  
Detailed mitigation and contingency for each milestone are provided in Section 4 below.

---

## 2. Algorithmic Complexity Assessment

### M0: Tensor Runtime & Core Types
- **Time**: `O(1)` per index-to-offset conversion; `O(rank)` per shape validation.
- **Space**: `O(rank)` for `Shape` and `Layout` metadata; tensor data stored contiguously.
- **Realizability note**: No compute kernels. Alignment checks are bitwise. Lean4 proofs are pure functions over `List Nat`. **Feasible with negligible resource cost.**

### M1: Clifford Algebra Core
- **Geometric product**: naive CPU reference is `O(bladeCount^2)` per multiplication.
  - For `Cl<3,0,0>`: `bladeCount = 8`, product cost = 64 muls/adds per product.
  - For `Cl<8,0,0>` (future, M6 benchmark): `bladeCount = 256`, product cost = 65,536 ops.
- **Rotor sandwich**: 2 geometric products + 1 multivector pass = `O(bladeCount^2)`.
- **Product table generation**: compile-time for hardcoded `Cl<3,0,0>`; `O(2^(3p+3q+3r))` symbolic enumeration if generic.
- **Realizability note**: M1 scope is hardcoded `Cl<3,0,0>`. 8-dimensional algebra fits in SIMD registers. **Feasible.** Generic compile-time tables are deferred post-M1.

### M2: HRM Backbone
- **Self-attention**: `O(B * T^2 * hidden)` per transformer block.
- **PrefixLM mask**: adds no compute cost beyond masking logits (elementwise).
- **Packed sequences**: indexing overhead `O(batch)` per partition check.
- **RoPE**: `O(T * hidden)` sine/cosine application.
- **H/L recurrence**: `H_cycles * L_cycles` transformer passes. With small cycles (2x2, 4x4) this is a constant factor < 16x over a standard transformer.
- **Realizability note**: Standard algorithms, well-understood complexity. **Feasible.**

### M3: HDIM Layer
- **Projection**: linear layer `O(B * T * hidden * heads * bladeCount)`.
  - FLOP estimate for config `n_layers=24, hidden_size=1280, num_heads=10, expansion=4`: forward pass ≈ `B * T * (24 * (4 * 1280² + 3 * 1280 * 1280/10))` base transformer FLOPs + Clifford overhead TBD after M1 micro-benchmark.
- **Invariant extraction**: `O(B * T * heads * bladeCount^2)`.
  - With blades=8: 64 muls per head → negligible vs attention.
- **Transfer**: same cost as extraction.
- **Fusion**: linear + elementwise gated residual = `O(B * T * hidden * heads * bladeCount)`.
- **Realizability note**: HDIM compute is additive to HRM but with small blade count (8) it is sub-dominant to attention. **Feasible.**

### M4: MSA Sparse Memory (CPU reference)
- **Routing key compute**: one HDIM extraction per query = `O(T_q * heads * bladeCount^2)`.
- **Scoring against N slots**: `O(N * k * heads * bladeCount)` where k = query token count per routing decision.
  - CPU reference with N=100 slots: trivial.
  - At scale (N=1M slots): requires approximate nearest neighbor, not brute force.
- **Top-k**: `O(N log k)` per query. For N=100, negligible.
- **Sparse attention over selected slots**: `O(B * T_q * k * slot_len * hidden)`.
- **Realizability note**: M4 delivers CPU reference with N <= 1000 slots. **Feasible as specified.** 100M-token production routing requires M6 custom kernels + vector index (HNSW/SCaNN) — this is explicitly deferred.

### M5: Composite Loss
- **L_CE**: standard cross-entropy `O(B * T * vocab)`.
- **L_aux**: contrastive loss over router weight pairs = `O(B * k^2)`.
- **L_iso**: squared norm over invariants = `O(B * heads * bladeCount)`.
- **MagicNorm-Clifford**: grade-wise norm = `O(params * bladeCount)`.
- **Realizability note**: Losses are additive; no new algorithmic complexity class. **Feasible.** Risk is hyperparameter tuning, not implementation.

### M6: CUDA-Oxide Kernels
- **Geometric product kernel**: `O(bladeCount^2)` per thread if fully parallel over B*T*heads.
  - Memory bound if bladeCount=256 (256 f32s = 1KB; fits in shared memory).
- **Sparse attention kernel**: memory-bound due to irregular K/V access from host RAM.
- CUDA kernel fusion target: benchmark against CPU reference after M6. No validated speedup claim yet.
  - 65K FLOPs per product; A100 peak = 312 TFLOPS (FP32). Roofline model is pending M6 benchmark validation.
  - 1KB per multivector * 2 operands + 1 result = 3KB per product. Memory bandwidth bound if throughput < ~500M products/s.
- **Realizability note**: Achievable if `cuda-oxide` submodule compiles. **Conditional on build pipeline.** Clifford kernel correctness (bit-exact vs CPU) is the hard part; numeric tolerance `ε = 1e-4` is generous for f32.

---

## 3. Critical Path Analysis

Dependency graph and milestone relationships are defined in `implementation_plan.md`.  
Time estimates below assume a single engineer.

| Path | Duration | Bottleneck |
|------|----------|------------|
| Training (M0 → M1 → M3 → M5) | 7 weeks | M2/M4 interfaces must freeze before M3/M5 integration |
| Inference scaling (M0 → M2 → M4 → M6) | 8+ weeks | M6 `cuda-oxide` build pipeline |

M3 and M2 can be developed in parallel after M0. M5 depends on M2+M3+M4 interfaces (mockable). M6 depends on M1+M4.

---

## 4. Risk Analysis by Milestone

Per-milestone risk levels (Low / Medium / High / Very High) are defined in `implementation_plan.md`.  
This section provides unique mitigation and contingency details not duplicated there.

### M0: Tensor Runtime — Risk: Low
- **Threat**: Lean4 `index_to_offset` / `offset_to_index` proof brittleness on `List Nat` recursion.
- **Mitigation**: Keep proofs small; use `simp` and `nlinarith` as in current `CoreTypes.lean`. No external dependencies.
- **Contingency**: If `lake build` becomes slow, split `CoreTypes.lean` into `Shape.lean` + `Index.lean`.

### M1: Clifford Core — Risk: Medium
- **Threat**: Generic `Cl<p,q,r>` compile-time product table requires proc-macros or const-generics beyond Rust stable capabilities.
- **Mitigation**: **M1 scope is explicitly hardcoded `Cl<3,0,0>`** (8 blades). Table of 64 entries can be hand-written or generated by a build script.
- **Contingency**: Defer generic signatures to post-M6. Training with 8 blades is sufficient for proof-of-concept.
- **Numeric threat**: Rotor unit validation `R * R^-1 = 1` may exceed tolerance due to normalization drift.
- **Mitigation**: Re-normalize rotor after each update; use `f64` for validation, `f32` for compute.

### M2: HRM Backbone — Risk: Low
- **Threat**: PrefixLM packed-sequence indexing errors (bidirectional prefix + causal suffix in same batch).
- **Mitigation**: Unit test mask generation with small tensors (B=2, T=8); verify no prefix attends to suffix.
- **Contingency**: If packed sequence complexity grows, implement simple left-padding first, optimize later.

### M3: HDIM Layer — Risk: Medium
- **Threat**: Hidden-size to blade-count mapping mismatch (e.g., hidden=768 does not divide evenly into heads*blades=64).
- **Mitigation**: Assert `hidden % (heads * bladeCount) == 0` at init; panic with clear message if violated. Use `HeadDivisible` contract from Lean.
- **Threat**: Domain rotor initialization produces non-unit rotors, breaking `unit_rotor_sandwich_identity`.
- **Mitigation**: Add `init_unit_rotor` constructor that normalizes and validates; fail fast at runtime if `UnitRotor` check fails.

### M4: MSA Sparse Memory — Risk: High (at scale)
- **Threat**: CPU-reference routing at 100M tokens is infeasible. Risk: M4 slot count > 10⁴ may exceed host DRAM bandwidth. Mitigation: benchmark before scale-up; fallback to smaller slot count or CPU routing. Acceptance: benchmark with ≤1000 slots / ≤100K tokens passes latency < 2× local-attention baseline and memory budget ≤ host DRAM capacity.
- **Mitigation**: **M4 scope is explicitly small-scale** (<=1000 slots, <=100K tokens). Document that 100M scale requires M6 + vector search index.
- **Threat**: Document-wise RoPE position collision if slot lengths vary.
- **Mitigation**: Global offset formula `k * (G/P)` with G > max slot length guarantees separation if P divides G evenly. Assert at cache append time.

### M5: Composite Loss — Risk: Medium
- **Threat**: `L_aux` or `L_iso` dominates `L_CE`, destabilizing training.
- **Mitigation**: Start with `lambda_aux = lambda_iso = 0.01`; scale up if gradients are healthy. Monitor ratio `|grad_iso| / |grad_ce|`.
- **Threat**: `MagicNorm-Clifford` clips too aggressively, starving learning.
- **Mitigation**: Use adaptive threshold (percentile of recent gradient norms), not fixed constant.

### M6: CUDA Kernels — Risk: Very High
- **Threat**: `cuda-oxide` submodule build pipeline is complex. CUDA-oxide submodule uses independent crate versions. Integration risk: API mismatch between `cuda-oxide` and `tensor-runtime`. Acceptance: `cargo test -p tensor-runtime --features cuda` passes.
- **Mitigation**: Start with `vecadd`-style sanity kernel in `cuda-kernels` to validate build before Clifford ops.
- **Threat**: Clifford kernel numeric difference exceeds `ε = 1e-4` vs CPU reference.
- **Mitigation**: Use `f64` intermediate accumulation in kernel, cast to `f32` at write-back. Compare grade-wise, not coefficient-wise. Clifford coefficients may partially cancel during transfer; this is normal GA behavior. No special handling required unless norm drops below `ε_min`.
- **Contingency**: If M6 fails, training loop (M5) still works on CPU. Inference uses smaller batch sizes or CPU-only path.

---

## 5. Resource Estimates

### Compute (training)
Resource estimates are pending benchmark harness (M5). Target configuration: 24 layers, 1280 hidden size, 10 heads. See `docs/implementation_plan.md` for per-milestone memory and compute budgets.

### Memory
- `Cl<3,0,0>` multivector: 8 f32s = 32 bytes per head.
- HDIM projection: 512 * 8 * 8 = 32K parameters per layer.
- MSA K/V cache at 1K slots * 4K tokens * 512 hidden * 2 (K+V) * 2 bytes (f16) = ~8 GB host RAM.
- Development GPU fit is pending benchmark harness (M5).

### Personnel / Time
- M0-M3 (critical path to working forward pass): 5 weeks, 1 engineer.
- M4-M5 (loss + small-scale memory): +2 weeks, 1 engineer.
- M6 (CUDA kernels): +4 weeks, requires CUDA expertise. Parallelizable with M4-M5 if second engineer available.

---

## 6. Integration Tolerance

The architecture has been designed with **graceful degradation**:

1. **M6 failure does not break M0-M5**. CPU reference is always available; CUDA is a backend dispatch option (`cpuReference` vs `cudaOxide`).
2. **M4 failure does not break M5**. MSA is optional for training; HRM+HDIM alone produce logits and losses.
3. **M1 generic signature failure does not block M3**. Hardcoded `Cl<3,0,0>` satisfies all M3 contracts.
4. **Lean4 proof failure does not block Rust implementation**. Rust code proceeds with runtime assertions; Lean catches up in parallel.

This tolerance is a deliberate architectural decision (see `TensorRuntime` backend enum and `HDIMHRMBridge` interface boundaries) and confirms realizability under partial failure.

---

## 7. Conclusion

The HAGI architecture is **realizable** under the stated milestone constraints:
- Hardcoded `Cl<3,0,0>` for M1.
- Small-scale MSA (<=1K slots) for M4.
- `cuda-oxide` integration as optional backend for M6.

The critical path to a training loop (M0-M5) is low-to-medium risk and can be completed in approximately 7 weeks by a single engineer. High-risk items (M4 at scale, M6) are isolated on a secondary path and have defined fallbacks.

**Recommended sequencing**: see `implementation_plan.md` Dependency Graph and Milestones.
