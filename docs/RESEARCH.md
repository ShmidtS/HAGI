# HAGI Research Background

This document summarizes the literature that informs HAGI's design and classifies the evidence strength for each claim. Evidence ratings:

- **Proven** — multiple published papers, reproducible results at scale
- **Promising** — published results, limited scale or conditions
- **Weak** — single paper, unreproduced, or theoretical
- **Marketing** — company claim, no independent validation
- **None** — no published evidence for this specific application

---

## The Core Problem: Reasoning Depth in Small Models

Reasoning capability in language models comes from three sources, in order of impact:

1. **Training data quality and scale** (dominant). Phi-4-mini (3.8B) matches 7B models through curated synthetic reasoning data, not architecture. At <10B parameters, data matters ~3× more than architecture.
2. **Parameter count** (second). More parameters = more stored knowledge. No architecture replaces this.
3. **Architecture** (third — but the only lever a solo developer fully controls). Determines how efficiently parameters are used.

The specific bottleneck HAGI attacks: **sequential reasoning depth.** A 12-layer model gets ~12 steps of computation per forward pass. Reasoning is compositional and needs more steps than small models have layers.

---

## Recurrent-Depth Transformers

**Evidence: Proven.**

Looping a shared transformer block gives N× effective depth at 1× parameter cost.

- **Universal Transformer** (Dehghani et al., 2019) — original parameter-shared recurrence with adaptive halting. +0.9 BLEU on WMT14.
- **Huginn-3.5B** (Geiping et al., 2025) — Prelude/Loop/Coda structure. An 8-layer physical model behaves like a 132-layer virtual model at 132 unrolls. Latent reasoning in continuous space, no explicit CoT tokens.
- **LoopLM** (Zhu et al., 2025) — looped pretraining at billion scale, production-viable.
- **RingFormer** — matches standard transformers with ~20% of parameters.

**Key limitation (important for HAGI):** Huginn shows gains from increasing recurrence are *modest* and plateau after ~8-16 iterations. On GSM8K, recurrence helps but does not match explicit chain-of-thought. **This plateau is the gap HAGI's grade decomposition targets** — flat recurrence has diminishing returns because all hidden dimensions converge at the same rate.

References:
- [Huginn / Scaling Test-Time Compute with Latent Reasoning](https://openreview.net/forum?id=S3GhJooWIC)
- [Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer](https://arxiv.org/html/2507.02199v1)
- [Looped Transformer Architectures](https://www.emergentmind.com/topics/looped-transformer-architectures)

---

## Clifford / Geometric Algebra in Neural Networks

**Evidence: Promising for geometry/vision. None for language.**

Geometric algebra represents data as multivectors with graded structure (scalar, vector, bivector, ...). The geometric product `uv = u·v + u∧v` simultaneously captures similarity (inner product) and oriented relational structure (wedge product).

- **GATr** (Brehmer et al., 2023) — Geometric Algebra Transformer for 3D/physics. Equivariant, 16-dim projective GA. Works for geometric tasks.
- **CGENNs** (Ruhe et al., NeurIPS 2023) — Clifford Group Equivariant Neural Networks. Grade projections are equivariant. Demonstrates per-grade structure is meaningful.
- **CliffordNet** (Jan 2026) — vision backbone on pure geometric algebra. 1.4M params match ResNet-18 (11.2M) on CIFAR-100. Uses geometric product as unified mixing+memory mechanism. **Vision only — explicitly not NLP.**

**The HAGI bet:** No published work uses Clifford grade structure to control *recurrence dynamics* in a language model. The CGENN result (grade projections carry distinct, meaningful information) plus the recurrence plateau (flat iteration has diminishing returns) motivate the hypothesis that per-grade update rates could extend the useful recurrence range. **This is unvalidated and is the primary research risk.**

References:
- [Geometric Algebra Transformer](https://arxiv.org/pdf/2305.18415)
- [Clifford Group Equivariant Neural Networks](https://arxiv.org/abs/2305.11141)
- [CliffordNet: All You Need is Geometric Algebra](https://arxiv.org/abs/2601.06793)

---

## Efficient Attention / KV Cache Compression

**Evidence: Proven.**

- **Multi-head Latent Attention (MLA)** — DeepSeek-V2/V3. Compresses KV into a low-dim latent (dim 512 in V3), >90% KV cache reduction with minimal quality loss. Deployed in production (671B params, 37B active).
- **Native Sparse Attention (NSA)** — DeepSeek, ACL 2025 Best Paper. Three parallel branches (compressed / selected / sliding) with learned gating. Hardware-aligned, natively trainable. Speeds comparable to FlashAttention-2.
- **GQA** — grouped-query attention. 3× KV cache reduction, negligible quality loss. Standard at all scales.

**For HAGI:** GQA from the start. MLA at Stage 3 for deployment. NSA deferred — not relevant to a 4K-context reasoning prototype.

References:
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Native Sparse Attention](https://arxiv.org/html/2502.11089v1)

---

## State-Space / Linear Recurrence Models

**Evidence: Mamba-2 Proven; RWKV-7 Promising.**

- **Mamba-2** — linear-time SSM. 2-8× training speedup, competitive benchmarks. Mamba2-2.7B: 39.6% MMLU vs Pythia-2.8B 36.5%. **Weaker than attention on recall-heavy tasks.**
- **RWKV-7 "Goose"** (March 2025) — Generalized Delta Rule, claims to exceed TC0 expressivity bound. SoTA among 3B open models on less training data. Constant memory, no KV cache.

**For HAGI:** Not the primary path. HAGI prioritizes reasoning (where attention leads), not inference speed. Considered as an alternative baseline only.

References:
- [Mamba SSM](https://github.com/state-spaces/mamba)
- [RWKV-7 "Goose"](https://arxiv.org/pdf/2503.14456)

---

## Hybrid Architectures

**Evidence: Promising.**

- **Jamba** (AI21, 2024) — hybrid Transformer-Mamba-MoE. 1:7 attention-to-Mamba ratio, MoE every other layer. 256K context with only 4GB attention cache (vs 32GB Mixtral). Only model with effective 256K on RULER.

**For HAGI:** Informative for long-context efficiency. Not adopted in the prototype — adds confounds.

References:
- [Jamba: Hybrid Transformer-Mamba](https://arxiv.org/abs/2403.19887)

---

## Memory-Augmented Models

**Evidence: Weak (early research).**

- **Titans** (Google, Jan 2025) — neural long-term memory module, learns to memorize at test time via momentum + forgetting. Claims 2M+ token sequences. Independent reimplementations exist but raise questions (see "Titans Revisited", Oct 2025).
- **TTT** (Test-Time Training) — updates weights at inference.

**For HAGI:** Deferred entirely. Too early, too complex, layers a second research project on the first.

References:
- [Titans: Learning to Memorize at Test Time](https://huggingface.co/papers/2501.00663)
- [Titans Revisited (critical analysis)](https://arxiv.org/pdf/2510.09551)

---

## Adaptive Computation / Halting

**Evidence: Proven (mechanism), Promising (at scale).**

- **PonderNet** (Banino et al., 2021) — probabilistic halting via geometric distribution. More stable than ACT.
- **PALBERT** — adapts PonderNet to stacked transformers.

**For HAGI:** Optional addition at Stage 1b (learned halting on the reasoning loop). Lets the model spend more iterations on harder tokens.

References:
- [PonderNet: Learning to Ponder](https://www.emergentmind.com/topics/pondernet)

---

## Low-Bit / Quantized Architectures

**Evidence: Proven for base LM; reasoning degradation real.**

- **BitNet b1.58** (Microsoft) — 2B params, 4T tokens, natively-trained 1.58-bit. Matches fp16 LLaMA in perplexity at 3B, 3.55× less memory, 2.71× faster. Requires custom training infrastructure.
- **PrismML Bonsai** (2026) — 1-bit / ternary edge models. **Marketing claim**: "intelligence density" metric is self-defined. Own numbers show MMLU-Pro 65.7 vs Qwen3-8B 83 — an 18-point reasoning gap from aggressive quantization. Not independently validated.

**For HAGI:** Train bf16, quantize post-training (Stage 4). The reasoning cost of 1-bit native training is the wrong tradeoff for a reasoning-focused model.

References:
- [BitNet b1.58-2B-4T](https://arxiv.org/pdf/2504.12285)
- [PrismML Bonsai analysis](https://computertech.co/prismml-bonsai-8b-review/)

---

## Small Efficient Models (Training Recipe Lessons)

**Evidence: Proven.**

- **Phi-4-mini** (3.8B) — high-quality synthetic data, matches 2× larger models on math/code. Reasoning variant: mid-training on distilled long-CoT → SFT → DPO → RL with verifiable reward.
- **SmolLM-3** (3B) — beats Llama-3.2-3B and Qwen2.5-3B through training recipe. Full engineering blueprint published.
- **CoT Curriculum Distillation** — 770M T5 reaches 94% of 540B teacher on SVAMP.

**For HAGI:** The dominant lesson — **training data quality > architecture at small scale.** Use CoT distillation from a frontier teacher. Generate synthetic relational-reasoning data specifically to test the grade-decomposition hypothesis.

References:
- [Phi-4-Mini-Reasoning](https://arxiv.org/abs/2504.21233)
- [CoT Curriculum Distillation](https://dl.acm.org/doi/10.1145/3775073.3775200)

---

## Long-Context Claims (Treat With Caution)

**Evidence: Marketing.**

- **SubQuadratic 12M context** (May 2026) — startup, 3 benchmarks published (RULER, MRCR v2, SWE-Bench), no general reasoning/math/safety eval, no independent validation.

**For HAGI:** Context length is not intelligence. GSM8K problems are <500 tokens. Ignore until validated.

References:
- [SubQuadratic 12M context launch](https://thenewstack.io/subquadratic-12-million-context-window/)

---

## Summary: What HAGI Adopts

| Technique | Stage | Evidence | Role |
|-----------|-------|----------|------|
| Recurrent depth | 1 | Proven | Foundation for reasoning |
| Grade-decomposed recurrence | 2 | None (the bet) | Core novel contribution |
| GQA | 0 | Proven | Standard efficiency |
| CoT distillation | 0 | Proven | Training data quality |
| MLA | 3 | Proven | Deployment efficiency |
| PonderNet halting | 1b | Proven | Adaptive compute |
| Post-training quantization | 4 | Proven | Local deployment |
| MoE | — | Proven at scale | Deferred (no benefit <1B) |
| Titans/TTT memory | — | Weak | Deferred |
| SSM base | — | Proven | Not chosen (reasoning priority) |
| 1-bit native training | — | Proven w/ caveats | Rejected (reasoning cost) |
