# HAGI Lean Architecture Rationale

## Introduction

HAGI unifies three architectural lines into a single system: HRM provides hierarchical reasoning via recurrent high-level/low-level states, HDIM adds structural analogy transfer across domains through Clifford algebra, and MSA extends accessible context via a sparse memory mechanism for documents and active queries. In this composition HRM handles step-by-step reasoning dynamics, HDIM extracts domain-independent structure, and MSA scales memory to long-context mode without full dense attention over all tokens.

The systemic idea: the HRM hidden state does not only pass through the transformer stack but also receives a structural channel. This channel projects the hidden representation into multivector `G` in `Cl(p,q,r)`, extracts invariant `U = R⁻¹ ⊗ G ⊗ R`, matches it with other domains, and transfers it back as a gated residual. Memory slots are treated as additional HDIM domains in the formal model; routing uses the same `DomainRotor` mechanism.

## Target Architecture

HAGI is built as a Rust-first runtime.

- The Rust workspace remains the primary production contour: tensor runtime, models, data/config, losses, FFI boundary, and future CUDA-oxide kernels.
- Milestone 0-1 must provably maintain workspace contracts and a forward-only CPU reference path: shape safety, layout alignment, correct backend dispatch boundaries, HRM recurrence shape preservation, HDIM transfer contract, MSA append-only cache.

## Key Components and Roles

### HRM Engine

The HRM Engine implements hierarchical reasoning via two states: `z_H` for high-level dynamics and `z_L` for low-level detail. Parameters `H_cycles`, `L_cycles`, `bp_steps`, `bp_warmup_ratio` set the depth and recurrence schedule. For the text path HRM uses PrefixLM packing, gated attention, RoPE, SwiGLU FFN, and a transformer stack.

- `HRMState` with `z_H` and `z_L`;
- H/L transition functions;
- preservation of `z_H` and `z_L` shape across cycles;
- PrefixLM mask legality: bidirectional prefix, causal suffix;
- packed sequence partition invariants;
- monotonic recurrence depth via `CycleId`.

### HDIM Core

The HDIM Core implements the Clifford algebra pipeline:

1. encode hidden/text/embedding into multivector `G`;
2. apply domain rotor `R`;
3. extract invariant `U = R⁻¹ ⊗ G ⊗ R`;
4. perform analogy match by structural invariant;
5. transfer structure into the target domain `G_B = R_B ⊗ U ⊗ R_B⁻¹`;
6. return the signal to HRM via gated fusion.

`CliffordOps`, `DomainRotor`, `UnitRotor`, `extractInvariant`, `domainTransfer`, `SameStructure`, `CrossDomainInvariantContract`, and the bridge between HRM hidden tensor and HDIM multivector.

### MSA Adapter

The MSA Adapter connects sparse memory for long-context inference. Architecturally it splits into three stages:

1. offline global memory encoding;
2. online routing/context assembly;
3. sparse generation.

K/V content pages are stored in host DRAM; routing keys `K̄ᵣ` are GPU-resident shards. The transfer contract is async fetch via CUDA stream ordering. In HAGI every memory slot receives a `DomainId`, so sparse memory becomes not just a retrieval source but a set of structural domains for HDIM transfer.

- memory slot as domain;
- route-within-slots invariant;
- document-wise RoPE separation;
- global RoPE for active query;
- append-only K/V cache monotonicity.

### Tensor Runtime

The Tensor Runtime defines the CPU/reference tensor API and boundary for future CUDA-oxide kernels. Its role is to keep a minimal stable ABI/contract between models and backend implementations.

- `TensorSpec`: shape, dtype, layout;
- alignment and stride invariants;
- backend dispatch safety;
- binary operation shape safety;
- preservation spec on the CPU reference ↔ CUDA-oxide transition.

### Lean4 Verification Layer

The Lean4 Verification Layer is a separate Lake package in `formalization/`. It formalizes core invariants without depending on the Rust compiler. Its task is to serve as an executable mathematical specification against which Rust crates can check API contracts.

## Integration Points

### HRM hidden state → HDIM multivector

`hrm-model` emits a hidden tensor. `hdim-model` applies projection `TensorSpec → Multivector`, where the hidden state shape must be compatible with the Clifford signature and coefficient representation dimension. At the Lean level this is specified by `HiddenProjection` and the theorem `preservesSignature`.

### HDIM invariant → HRM gated residual

After `extractInvariant` and optional `domainTransfer` the result returns to the HRM path through `GatedFusion`. Fusion is not allowed to change the hidden tensor shape. At the Lean level this is the contract `GatedFusion.preservesShape` and `bridge_preserves_hidden_shape`.

### MSA memory slots → HDIM domains

Every MSA memory slot receives a `DomainId`. HDIM invariant extraction applies to memory-slot content after routing selection. The selected slot's domain rotor is used for transfer into the active query domain. This creates a shared mechanism for cross-document structural analogy transfer: the active query is one domain, the memory document is another.

### Tensor Runtime boundary

HRM, HDIM, and MSA operate only through the `TensorSpec`/runtime boundary. Backend dispatch must not change the shape/dtype/layout contract. CUDA-oxide kernels are required to preserve formal semantics. CPU-vs-CUDA parity tests with tolerance `ε` are the acceptance gate; equivalence is not yet proven for all kernels.

## Lean4 Formalization Roadmap

### Stage 0: Workspace contracts

- `CoreTypes.lean`: shape, dtype, domain/cycle IDs, layout invariants.
- `TensorRuntime.lean`: backend dispatch safety, alignment, shape-preserving ops.
- Goal: formal minimum for the CPU reference path.

### Stage 1: HRM forward-only path

- `HRM.lean`: `HRMState`, H/L transitions, shape preservation.
- PrefixLM mask legality.
- Packed sequence partition invariants.
- Depth monotonicity for recurrence.

### Stage 2: HDIM algebraic contracts

- `HDIM.lean`: Clifford signature, multivector, rotor unit predicate.
- `extractInvariant`, `domainTransfer`, `SameStructure`.
- Cross-domain invariant contract.
- HRM projection/fusion bridge.

### Stage 3: MSA integration

- `MSA.lean`: memory slot as domain.
- Sparse routing safety.
- Document-wise/global RoPE separation.
- K/V cache append-only theorem.

### Stage 4: System invariants

- `Invariants.lean`: end-to-end forward pass shape preservation.
- HDIM extraction+transfer identity modulo target domain.
- HRM recurrence depth monotonicity.
- Memory cache append-only invariant.

### Stage 5: Rust conformance

- Map Rust structs to Lean specs by naming and contract tests.
- Add property tests in Rust for Lean-shaped invariants.
- Keep Lean proofs abstract where numerical Float equality would be unsound or backend-dependent.
