# HAGI Research Background

This document summarizes the literature that informs HAGI's design and classifies the evidence strength for each claim. Evidence ratings:

- **Proven** -- multiple published papers, reproducible results at scale
- **Promising** -- published results, limited scale or conditions
- **Weak** -- single paper, unreproduced, or theoretical
- **Marketing** -- company claim, no independent validation
- **None** -- no published evidence for this specific application

---

## The Core Problem: Reasoning Depth in Small Models

Reasoning capability in language models comes from three sources, in order of impact:

1. **Training data quality and scale** (dominant). Phi-4-mini (3.8B) matches 7B models through curated synthetic reasoning data, not architecture. At <10B parameters, data matters ~3x more than architecture.
2. **Parameter count** (second). More parameters = more stored knowledge. No architecture replaces this.
3. **Architecture** (third -- but the only lever a solo developer fully controls). Determines how efficiently parameters are used.

The specific bottleneck HAGI attacks: **sequential reasoning depth.** A 12-layer model gets ~12 steps of computation per forward pass. Reasoning is compositional and needs more steps than small models have layers.

---

## Recurrent-Depth Transformers

**Evidence: Proven.**

Looping a shared transformer block gives Nx effective depth at 1x parameter cost.

- **Universal Transformer** (Dehghani et al., 2019) -- original parameter-shared recurrence with adaptive halting. +0.9 BLEU on WMT14.
- **Huginn-3.5B** (Geiping et al., 2025) -- Prelude/Loop/Coda structure. An 8-layer physical model behaves like a 132-layer virtual model at 132 unrolls. Latent reasoning in continuous space, no explicit CoT tokens.
- **LoopLM** (Zhu et al., 2025) -- looped pretraining at billion scale, production-viable.
- **RingFormer** -- matches standard transformers with ~20% of parameters.

**Key limitation (important for HAGI):** Huginn shows gains from increasing recurrence are *modest* and plateau after ~8-16 iterations. On GSM8K, recurrence helps but does not match explicit chain-of-thought. **This plateau is the gap HAGI's grade decomposition targets** -- flat recurrence has diminishing returns because all hidden dimensions converge at the same rate.

References:
- [Huginn / Scaling Test-Time Compute with Latent Reasoning](https://openreview.net/forum?id=S3GhJooWIC)
- [Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer](https://arxiv.org/html/2507.02199v1)
- [Looped Transformer Architectures](https://www.emergentmind.com/topics/looped-transformer-architectures)

---

## Clifford / Geometric Algebra in Neural Networks

**Evidence: Promising for geometry/vision. None for language.**

Geometric algebra represents data as multivectors with graded structure (scalar, vector, bivector, ...). The geometric product `uv = u.v + u^v` simultaneously captures similarity (inner product) and oriented relational structure (wedge product).

- **GATr** (Brehmer et al., 2023) -- Geometric Algebra Transformer for 3D/physics. Equivariant, 16-dim projective GA. Works for geometric tasks.
- **CGENNs** (Ruhe et al., NeurIPS 2023) -- Clifford Group Equivariant Neural Networks. Grade projections are equivariant. Demonstrates per-grade structure is meaningful.
- **CliffordNet** (Jan 2026) -- vision backbone on pure geometric algebra. 1.4M params match ResNet-18 (11.2M) on CIFAR-100. Uses geometric product as unified mixing+memory mechanism. **Vision only -- explicitly not NLP.**

**The HAGI bet:** No published work uses Clifford grade structure to control *recurrence dynamics* in a language model. The CGENN result (grade projections carry distinct, meaningful information) plus the recurrence plateau (flat iteration has diminishing returns) motivate the hypothesis that per-grade update rates could extend the useful recurrence range. **This is unvalidated and is the primary research risk.**

References:
- [Geometric Algebra Transformer](https://arxiv.org/pdf/2305.18415)
- [Clifford Group Equivariant Neural Networks](https://arxiv.org/abs/2305.11141)
- [CliffordNet: All You Need is Geometric Algebra](https://arxiv.org/abs/2601.06793)

---

## Hierarchical Reasoning

**Evidence: Promising.**

HAGI's HRM implements two-level recurrent reasoning (strategic z_H + tactical z_L), inspired by hierarchical reinforcement learning:

- **HRL** (hierarchical RL) -- two-level control with strategic (high-level) and tactical (low-level) policies is well-established in RL.
- **HRM original design** -- HAGI's prior architecture used separate H/L transformer stacks. The current design replaces this with grade momentum (slow scalar = "H", fast bivector = "L") within a single shared block, avoiding architectural duplication.

**For HAGI:** HRM controls the reasoning loop depth (H_cycles x L_cycles). The H/L distinction is now achieved through grade momentum within GDR, not through separate parameter stacks.

---

## Domain-Invariant Transfer

**Evidence: Weak (theoretical motivation).**

HDIM's rotor sandwich extracts domain-invariant structure from hidden states and transfers it across domains. The theoretical basis:

- **Clifford rotors** -- even-grade multivectors that preserve norms under the geometric product. `R^-1 * G * R` extracts the invariant component.
- **CGENN equivariance** -- grade projections are equivariant under rotor actions, meaning the invariant extraction is mathematically well-defined.

**For HAGI:** HDIM applies this to language model hidden states. The "domains" are different reasoning iterations (or different MSA memory slots). The hypothesis: invariant structure extracted in one iteration can be transferred to improve the next.

---

## Memory-Augmented Attention

**Evidence: Promising (for sparse retrieval); Weak (for HDIM routing).**

- **Native Sparse Attention (NSA)** -- DeepSeek, ACL 2025 Best Paper. Three parallel branches with learned gating.
- **DeepSeek-V2/V3 MLA** -- compresses KV into low-dim latent, >90% KV cache reduction.
- **Memorizing Transformers** -- kNN-augmented attention with external memory.

**For HAGI:** MSA uses slot-based sparse attention with HDIM-invariant routing keys. The slot registry is cleared every forward in training; at inference, a persistent registry accumulates across decode steps. The routing key is the Clifford scalar invariant, not a learned projection.

---

## Mixture of Experts

**Evidence: Proven at scale.**

- **Switch Transformer** (Fedus et al., 2021) -- top-1 expert routing, proven at billion-parameter scale.
- **Mixtral** -- 8x7B MoE, production-deployed.
- **Mixture-of-Depths** (Raposo et al., 2024) -- extra "skip" router slot lets trivial tokens bypass the MLP.

**For HAGI:** MoE SwiGLU with 4 experts (top-1) + MoD skip. At 74M parameters, MoE provides conditional computation without increasing parameter count -- each token activates only 1 of 4 experts, plus the skip option.

References:
- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258)

---

## Block-Wise Generation

**Evidence: Promising.**

- **Medusa** (Cai et al., 2024) -- multiple prediction heads for parallel token generation.
- **EAGLE** -- speculative decoding with draft model.

**For HAGI:** CAST predicts K=8 tokens per forward pass via Clifford multivector virtual states. The geometric product between adjacent predictions provides a structural coherence signal absent in Medusa-style independent heads.

---

## Reasoning Cache

**Evidence: Promising (single paper).**

- **Reasoning Cache** (Wu et al., 2026, arXiv:2602.03773) -- iterative generate-summarize-cache decoding. Exploits asymmetry between response generation and summarization capabilities. Effective reasoning horizon = T x (H_R + H_S), but each step operates on bounded context.

**For HAGI:** RC is integrated with MSA -- summary hidden states are registered as MSA slots, enabling cross-iteration memory retrieval through sparse attention.

References:
- [Reasoning Cache (arXiv:2602.03773)](https://arxiv.org/abs/2602.03773)

---

## Knowledge Distillation for Small Models

**Evidence: Proven.**

- **Phi-4-mini** (3.8B) -- high-quality synthetic data, matches 2x larger models on math/code. Reasoning variant: mid-training on distilled long-CoT -> SFT -> DPO -> RL.
- **SmolLM-3** (3B) -- beats Llama-3.2-3B and Qwen2.5-3B through training recipe.
- **CoT Curriculum Distillation** -- 770M T5 reaches 94% of 540B teacher on SVAMP.

**For HAGI:** SmolLM2-135M serves as teacher. Embedding transfer at init (exact copy) + KL divergence on soft logits during training. Teacher freed at 60% training to reclaim VRAM.

References:
- [Phi-4-Mini-Reasoning](https://arxiv.org/abs/2504.21233)
- [CoT Curriculum Distillation](https://dl.acm.org/doi/10.1145/3775073.3775200)

---

## RL for Reasoning (MGPO)

**Evidence: Promising.**

- **GRPO** (Group Relative Policy Optimization) -- used in DeepSeek-R1, proven for reasoning RL.
- **VibeThinker-3B** -- MGPO variant with prompt difficulty weighting.
- **MaxEnt guidance** -- `w(q) = exp(-gamma * |p(q) - p0|)` focuses updates on prompts near the model's capability boundary.

**For HAGI:** MGPO adapted for single-GPU 8GB: group size 4, sequential rollouts, gradient checkpointing during update. Optional Long2Short reward shift for brevity.

References:
- [VibeThinker-3B (MGPO)](https://arxiv.org/abs/2503.02293)

---

## Efficient Attention / KV Cache Compression

**Evidence: Proven.**

- **GQA** -- grouped-query attention. 3x KV cache reduction, negligible quality loss. Standard at all scales.
- **MLA** -- DeepSeek-V2/V3. Compresses KV into low-dim latent.
- **INT8 KV cache** -- quantize K/V to int8 with per-head fp16 scales. 2x cache reduction.

**For HAGI:** GQA (8 query, 4 KV heads) from the start. INT8 KV cache at inference. RoPE theta 500000 for extended extrapolation beyond the 1024 training window.

---

## Muon Optimizer

**Evidence: Promising.**

- **Muon** (Keller Jordan) -- Newton-Schulz orthogonalization of gradient updates for 2D weight matrices. Scale-invariant: higher LR converges faster without grad clipping.
- **Newton-Schulz quintic** -- 5-iteration approximation of `G(G^T G)^{-1/2}`. Converges well for 2D hidden weights.

**For HAGI:** Muon for 2D weights (attention, MLP, GDR-MLP) + AdamW for 1D (embeddings, norms, gates). Scale-aware weight decay (0.5) bounds weight norms at `||W||_ss ~ 2.0`, matching the residual-scaled init. This structurally prevents the residual-stream divergence that occurred with unbounded Muon weight norms.

References:
- [Muon optimizer (Keller Jordan)](https://github.com/KellerJordan/Muon)

---

## Non-Axiomatic Reasoning System (NARS)

**Evidence: Weak (theoretical framework, no ML integration published).**

- **OpenNARS** -- non-axiomatic logic system with truth revision, budget allocation, and bag-based concept selection.
- **Truth values** -- (frequency, confidence) tuples, revised via an algebraic rule.
- **Budget values** -- priority, durability, quality -- decay over time.

**For HAGI:** NARS controllers observe training signals (loss, gradient norms) and dynamically adjust HRM cycle counts, HDIM domain transfer, and MSA slot routing. Disabled by default; the integration is experimental.

See [NARS.md](NARS.md) for detailed documentation.

References:
- [OpenNARS](https://github.com/opennars/opennars)

---

## Summary: What HAGI Adopts

| Technique | Evidence | Role |
|-----------|----------|------|
| Recurrent depth (HRM) | Proven | Foundation for reasoning |
| Grade-decomposed recurrence (GDR) | None (the bet) | Core novel contribution |
| Domain-invariant transfer (HDIM) | Weak | Cross-iteration structure transfer |
| Memory sparse attention (MSA) | Promising | Extended context via external memory |
| MoE + MoD | Proven | Conditional computation |
| CAST block generation | Promising | 8x faster generation |
| Reasoning Cache | Promising | Extended reasoning horizon |
| Knowledge distillation | Proven | Embedding transfer + soft targets |
| MGPO RL | Promising | Reasoning RL |
| Muon optimizer | Promising | Faster convergence, scale-invariant |
| GQA + INT8 KV | Proven | Standard efficiency |
| NARS controllers | Weak | Experimental adaptive control |
| fp16 attention | Proven | Better softmax resolution |
