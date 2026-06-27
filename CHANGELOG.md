# Changelog

All notable changes to HAGI are documented here.


## [Unreleased]

### Added -- Architecture

- **GDR (Grade-Decomposed Recurrence)**: Clifford algebra `Cl(3,0,0)` grade-structured hidden state with per-grade update dynamics (scalar momentum 0.8, vector 0.5, bivector/trivector full update). Geometric product cross-grade interaction. Learnable GDR capacity router (MoE-style). `src/hagi/model/gdr.py`
- **HRM (Hierarchical Recurrent Model)**: Two-level reasoning loop (H-cycles x L-cycles) with strategic (z_H, 160-dim) and tactical (z_L, 160-dim) states. Memory-aware HRM: MSA read+write inside L-cycle loop. Stochastic depth (0.3) and progressive reasoning budget (30K steps). `src/hagi/model/hrm_full.py`
- **HDIM (Hidden-state Decomposed Invariant Module)**: Full pipeline: project -> invariant extraction (rotor sandwich) -> domain transfer -> gated fusion. 4 parallel domain rotors with LCG-scheduled selection. Delayed HDIM aggregation. `src/hagi/model/hdim_full.py`
- **MSA (Memory Sparse Attention)**: Slot-based sparse attention with HDIM-invariant routing. Slot registry (4096 cap) with append-only K/V cache. Chunk compression (4 tokens/slot). Adaptive top_k (MoD-guided). LSH routing (implemented, disabled for RTX 3070). `src/hagi/model/msa.py`
- **MoE (Mixture of Experts)**: SwiGLU expert routing (4 experts, top-1). Mixture-of-Depths skip router. Load-balance auxiliary loss. Router temperature. `src/hagi/model/moe.py`
- **CAST (Clifford Algebra Symbolic Reasoning Tokens)**: Block-wise generation predicting K=8 tokens per forward pass via multivector virtual states with geometric product coherence. Training subsampling (train_k=3) with exponential k-loss decay. `src/hagi/model/cast.py`
- **NARS (Non-Axiomatic Reasoning System)**: Optional OpenNARS-style controllers for HRM/HDIM/MSA. Truth revision, budget allocation, bag-based selection. Disabled by default. `src/hagi/nars/`
- **Reasoning Cache**: Iterative generate-summarize-cache decoding (arXiv:2602.03773). MSA integration for cross-iteration memory retrieval. `src/hagi/inference/reasoning_cache.py`
- **Binary Factorized Layers**: Experimental 1-bit +/-1 weights via STE. `src/hagi/model/binary_factorized.py`
- **KV Cache**: INT8 quantized KV cache with per-head fp16 scales. Dynamic and static cache implementations. `src/hagi/model/kv_cache.py`
- **Inference Optimizations**: Inference-optimized model wrapper. `src/hagi/model/inference_opt.py`

### Added -- Training

- **Canonical training script** (`scripts/train.py`, 1038 lines): Sequential cycling curriculum, VRAM probe, prefix-LM, full resume, dry-run mode.
- **Muon+AdamW hybrid optimizer**: Newton-Schulz quintic orthogonalization for 2D weights, AdamW for 1D. Scale-aware weight decay (0.5). Batched zero-power. Schedule-Free AdamW alternative. `src/hagi/train/optim.py`
- **Knowledge distillation**: SmolLM2-135M teacher. Embedding transfer at init. KL divergence with chunked logits (no [B,T,V] materialization). Teacher freed at 60% training. `src/hagi/train/distillation.py`
- **RL training (MGPO)**: MaxEnt-Guided Policy Optimization adapted from VibeThinker-3B. Group-relative advantage, prompt difficulty weighting, Long2Short reward shift. `scripts/train_rl.py`, `src/hagi/train/rl_loop.py`
- **Composite loss**: L_CE + L_iso + L_moe + L_msa_lb + L_gdr_router with warmup. Fused linear CE (no logits materialization). Label smoothing 0.05. `src/hagi/losses.py`
- **WSD schedule**: Warmup-Stable-Decay, better for small models with embedding transfer.
- **Sequential cycling curriculum**: Easy -> mid -> hard dataset ordering for 3 cycles.
- **EMA**: Exponential moving average of model parameters.
- **Profiling**: `scripts/profile_steps.py` with synthetic GPU-resident tokens.

### Added -- Inference

- **ChatSession**: Interactive chat REPL with history, streaming, LoRA, online learning, RC integration. `src/hagi/inference/chat.py`
- **Generation**: Standard, streaming, and rollout-based generation. VibeHarness Config/Decision patterns. `src/hagi/inference/generate.py`
- **LoRA adapters**: Low-rank adaptation with auto-application. `src/hagi/inference/lora.py`
- **Online learning**: Feedback-driven adaptation. `src/hagi/inference/online.py`
- **Chat CLI**: `scripts/chat.py` and `src/hagi/inference/cli.py`

### Added -- Data

- **Memmap packed format**: Binary memmap with BFD packing (uint16). `src/hagi/data/`
- **Sequential cycling**: `SequentialCyclingIterator` for curriculum ordering.
- **Prefix-LM**: Bidirectional prefix, causal suffix. `src/hagi/data/prefix_lm.py`
- **SFT dataset**: Instruction tuning support. `src/hagi/data/sft_dataset.py`
- **Tokenizer wrapper**: SmolLM2 tokenizer integration. `src/hagi/data/tokenizer.py`

### Added -- Evaluation

- **lm-eval-harness adapter**: HAGI model registration for standard benchmarks. `src/hagi/eval/lm_eval_wrapper.py`
- **Golden benchmarks**: Built-in benchmark suite. `src/hagi/eval/golden.py`
- **Intelligence-density metrics**: HAGI-IQ and HAGI-IPP. `src/hagi/eval/evaluate.py`

### Added -- Config

- **`configs/rtx3070_canonical.yaml`** (1138 lines): Single source of truth with inline rationale for every non-default value. `Read in:` tags cross-reference consuming code.
- **`configs/rl_rtx3070.yaml`**: RL training config.

### Added -- Tooling

- **`torch-compile.bat`**: MSVC wrapper for torch.compile on Windows.
- **`pyproject.toml`**: Editable install, basedpyright config.

### Changed

- Architecture pivoted from stacked HRM+HDIM+MSA+MoE+Titans (prior design) to integrated GDR+HRM+HDIM+MSA+MoE+CAST with controlled ablation.
- HRM H/L distinction reframed as grade momentum within a shared block (no architectural duplication).
- Clifford structure moved from bolted-on layer (prior HDIM) to integral recurrence mechanism (GDR).
- Hidden size reduced 768 -> 576 for RTX 3070 8GB VRAM budget.
- Layer split changed 4+4+4 -> 2+7+2 (reasoning deepened, perception/expression trimmed).
- Precision changed from bf16-autocast to manual_bf16 (OOM fix for 8GB).
- Optimizer changed from plain AdamW to Muon+AdamW hybrid.
- Training data expanded from FineWeb-Edu only to 9-source mix with curriculum.
- VRAM scaling projections added for 7GB-48GB GPUs.

### Performance Optimizations

- fp16 attention (QKV cast for 8x better softmax resolution)
- fp32 RMSNorm (exact variance computation)
- Fused QKV and gate-up projections (3x fewer kernel launches)
- Gradient checkpointing with group size 2 (balanced [2,2,2,1] split)
- torch.compile support
- INT8 KV cache (2x cache memory reduction)
- CAST block generation (8x fewer forward passes)
- Adaptive MSA top_k (25% attention reduction for trivial tokens)
- Stochastic HRM depth (15% compute savings)
- Progressive reasoning budget (10% training time savings)

### Removed

- Rust workspace (`crates/`) -- not in current branch. Will be re-added post-validation.
- Lean4 formalization (`formalization/`) -- not in current branch. Will be re-added post-validation.
- `prototype/` directory -- replaced by `src/hagi/` library.
- `docs/implementation_plan.md`, `docs/realizability_verification.md`, `docs/hagi-lean-architecture.md` -- Rust-specific docs, not applicable to current PyTorch-only architecture.
- `docs/WSL2-SETUP.md` -- WSL2 setup for Rust/CUDA, not needed for PyTorch.
- MoE/MoE, MSA, Titans from core experiment (prior design) -- MSA and MoE are back but as optional config flags, not mandatory stack.

## [0.1.0] -- Prior design (archived direction)

### Added
- Initial Rust workspace scaffold (10 crates).
- Lean4 formalization of core invariants.
- Architecture docs for HRM + HDIM + MSA design.
- cuda-oxide submodule integration.
- PyTorch prototype in `prototype/` with GDR model, training stack, evaluation adapter.
