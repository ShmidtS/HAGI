# HAGI — Hypercomplex Artificial General Intelligence

HAGI unites a hierarchical recurrent language model architecture ([HRM-Text](https://github.com/sapientinc/HRM-Text)) with a geometric structural layer of hypercomplex invariants (HDIM) in a single Rust system with CUDA kernels via [cuda-oxide](https://github.com/NVlabs/cuda-oxide).

---

## Concept

### HRM as Backbone
The Hierarchical Recurrent Model (HRM) splits computation into two levels:
- **L-module** — fast local refinement recurrence, responsible for token-level processing.
- **H-module** — slow global controller recurrence, manages abstract planning.

On each H-cycle the L-module performs several refinement iterations, after which the H-module integrates the result into the global hidden state.

### HDIM as Structural Layer
The Hypercomplex Domain Isomorphism Machine (HDIM) transforms H-module hidden states into Clifford-algebra multivectors. Key operations:
- **Sandwich product** extracts domain-invariant encoding: `U_inv = R^{-1} ⊗ G ⊗ R`.
- **Transfer** moves the invariant into the target domain: `G_target = R_target ⊗ U_inv ⊗ R_target^{-1}`.
- **Geometric product** encodes both similarity (scalar/inner) and relational structure (bivector/wedge).

This lets the model "think" not merely in vector space but in a geometrically rich structure where relations between concepts are governed by rotors and bivectors.

### MoE + Hierarchical Recurrence
Mixture-of-Experts (MoE) routes representations to specialized experts:
- Low-level experts — local computations of the L-module.
- Structural experts — geometric transformations of multivectors.
- Domain experts — rotors for cross-domain transfer.
- Memory experts — interaction with long-term context.

### Titans/TTT Memory
Adaptive Titans/TTT-style memory updates online through gradient steps on the surprise error. Combined with HRM hierarchical states, this creates powerful long-term memory with on-the-fly domain transfer capability.

---

## Why It Works

Vector spaces discard structural information: dot-product similarity cannot distinguish "A causes B" from "B causes A". Clifford multivectors preserve the directionality of relations through grade structure (vectors, bivectors, trivectors). HRM provides hierarchical context reprocessing, while HDIM supplies a structural invariant layer resilient to domain shift.

---

## Key Components

| Component | Rust Crate | Description |
|---|---|---|
| Clifford core | `clifford-core` | Clifford algebra, geometric product, rotors, norms, inverses |
| HRM model | `hrm-model` | H/L Transformer stacks, recurrent scheduler, PrefixLM attention |
| HDIM model | `hdim-model` | Hidden → multivector projection, sandwich extraction, domain transfer |
| MoE | `moe` | Router, dispatch/combine, Z-loss, expert orthogonalization |
| Memory | `memory` | Titans/TTT adaptive memory with online update |
| Losses | `losses` | CE + reconstruction + isomorphism + InfoNCE + routing + ortho + memory |
| Training | `training` | PrefixLM packing, multipack LPT, truncated BPTT, AdamATan2 |
| Data | `data` | Synthetic pair/triplet generation for contrastive losses |
| CUDA kernels | `cuda-kernels` | cuda-oxide kernels: attention, Clifford ops, MoE, memory |

---

## Current Status

The project is in the architectural design phase.


The first milestone (toy model, ~8 layers, 256 hidden, single-GPU forward parity) is in planning.

---

## License

[Apache License 2.0](LICENSE).
