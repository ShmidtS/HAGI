# HAGI Configuration Reference

`configs/rtx3070_canonical.yaml` is the **single source of truth** (1138 lines). Every non-default value has an inline comment documenting the rationale. This document provides a structured reference.

For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md). For training workflow, see [TRAINING.md](TRAINING.md).

---

## Config Sections

```
rtx3070_canonical.yaml
|-- data          # Data sources, tokenization, batching
|-- eval          # Benchmark evaluation
|-- model         # Architecture configuration
|-- training      # Optimizer, schedule, loss, loop
`-- rl            # RL training (MGPO)
```

---

## Data Section

| Key | Default | Description |
|-----|---------|-------------|
| `dataset_mode` | `memmap_packed` | Binary memmap with BFD packing |
| `max_seq_len` | 1024 | Maximum sequence length (tokens) |
| `min_seq_len` | 1024 | Minimum sequence length (fixed = densest signal) |
| `mix` | (see below) | Data mix weights (source -> proportion) |
| `mix_paths` | (see below) | Explicit .bin paths with curriculum ordering |
| `num_workers` | 0 | DataLoader workers (0 = synchronous) |
| `packing` | `bfd` | Best-Fit-Decreasing on EOS boundaries |
| `sequential_cycles` | 3 | Iterate datasets in order N times before shuffling |
| `pin_memory` | true | Pin memory for faster CPU->GPU transfer |
| `tokenizer` | `HuggingFaceTB/SmolLM2-135M` | Tokenizer model (vocab=49,152) |
| `train_tokens` | 3,000,000,000 | Total training tokens (derives max_steps) |
| `dtype` | `uint16` | Memmap dtype (supports vocab up to 65535) |

### Data Mix

| Source | Weight | Curriculum Phase |
|--------|--------|-----------------|
| edu | 0.40 | 3 (hard) |
| slimpajama | 0.25 | 3 (hard) |
| wikipedia_en | 0.10 | 2 (mid) |
| wikipedia_ru | 0.08 | 2 (mid) |
| oscar_ru | 0.07 | 3 (hard) |
| openwebmath | 0.03 | 2 (mid) |
| smoltalk | 0.03 | 1 (easy) |
| tinystories | 0.02 | 1 (easy) |
| python_instruct | 0.02 | 1 (easy) |

### Sequential Cycling

`sequential_cycles: 3` -- datasets iterated in difficulty order (easy -> mid -> hard) 3 times before shuffling. The list order in `mix_paths` IS the curriculum; weights are used only in mixed mode (sequential_cycles=0).

---

## Eval Section

| Key | Default | Description |
|-----|---------|-------------|
| `benchmarks` | gsm8k, arc_challenge, boolq, hellaswag, winogrande | lm-eval-harness benchmarks |

Note: eval section is read by `hagi.eval.evaluate` CLI, NOT by `scripts/train.py`. Training reads `eval_interval`/`eval_samples`/`eval_iters` from the `training` section.

---

## Model Section

### Cross-Entropy / Loss

| Key | Default | Description |
|-----|---------|-------------|
| `ce_chunk_size` | 0 | Chunk size for CE computation (0 = no chunking) |
| `use_fused_ce` | true | Fused lm_head + CE (logits is None in training) |
| `ce_fused_chunk_size` | 4096 | Chunk size for fused CE path |
| `label_smoothing` | 0.05 | Token CE label smoothing |

### CAST

| Key | Default | Description |
|-----|---------|-------------|
| `cast_config.block_size` | 8 | K-token block prediction |
| `cast_config.use_coherence` | true | Geometric product coherence between adjacent predictions |
| `cast_config.gate_init` | 0.0 | Coherence gate init (0.0 = sigmoid(-5) ~ 0.007, disabled at init) |
| `cast_config.train_k` | 3 | Number of K positions to compute CE loss on (subsample) |
| `cast_config.k_loss_decay` | 0.5 | Exponential decay weight per k position |

### Architecture: Layer Counts

| Key | Default | Description |
|-----|---------|-------------|
| `perception_layers` | 2 | Perception stage blocks (trimmed 4->2) |
| `reasoning_layers` | 7 | Reasoning stage blocks (deepened 4->7, HAGI's novelty) |
| `expression_layers` | 2 | Expression stage blocks (trimmed 4->2) |

### Core Dimensions

| Key | Default | Description |
|-----|---------|-------------|
| `hidden_size` | 576 | Hidden dimension (head_dim 72 = 576/8) |
| `vocab_size` | 49152 | SmolLM2 tokenizer vocab |

### Reasoning Loop

| Key | Default | Description |
|-----|---------|-------------|
| `loop_count` | 5 | Reasoning iterations (used when hrm=false) |
| `use_loop` | true | Enable reasoning loop |

### GDR (Grade-Decomposed Recurrence)

| Key | Default | Description |
|-----|---------|-------------|
| `use_gdr` | true | Enable Clifford grade-decomposed recurrence |
| `hdim_full` | true | Use full HDIM module (project -> invariant -> transfer -> fuse -> GDR) |
| `hdim_heads` | 4 | Parallel multivector projections |
| `hdim_num_rotors` | 4 | Parallel domain rotors |
| `hdim_delay_steps` | 1 | Delayed HDIM aggregation (1 = no delay) |
| `use_hdim_cross_domain` | true | Cross-domain transfer (rotor sandwich) |
| `rotor_seed` | 42 | Deterministic rotor schedule seed |

### Grade Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `grades.scalar` | 64 | Grade 0: confidence (slow, momentum 0.8) |
| `grades.vector` | 96 | Grade 1: entities (medium, momentum 0.5) |
| `grades.bivector` | 96 | Grade 2: relations (fast, full update) |
| `grades.trivector` | 64 | Grade 3: higher-order structure (fast, full update) |
| `grades.residual` | 256 | Unconstrained channel |
| `grades.scalar_momentum` | 0.8 | Scalar grade momentum (slow tier) |
| `grades.vector_momentum` | 0.5 | Vector grade momentum (medium tier) |
| `grades.gdr_router` | true | Learnable GDR capacity router (MoE-style) |
| `grades.gdr_router_alpha` | 0.01 | Router load-balance aux loss coefficient |
| `grades.gdr_router_temperature` | 1.0 | Router temperature |

### HRM (Hierarchical Recurrent Model)

| Key | Default | Description |
|-----|---------|-------------|
| `hrm` | true | Enable HRM two-level reasoning controller |
| `hrm_memory_aware` | true | MSA read+write inside L-cycle loop |
| `h_dim` | 160 | High-level state dimension (z_H) |
| `l_dim` | 160 | Low-level state dimension (z_L) |
| `hrm_h_cycles` | 1 | H-cycles per forward pass |
| `hrm_l_cycles` | 2 | L-cycles per H-cycle (14 reasoning passes/step) |
| `hrm_stochastic_depth` | 0.3 | Probability of skipping L-cycle 1 |
| `hrm_progressive_start_step` | 30000 | Use 1 L-cycle until this step, then full |

### MSA (Memory Sparse Attention)

| Key | Default | Description |
|-----|---------|-------------|
| `use_msa` | true | Enable slot-based sparse attention |
| `msa_slot_count` | 4096 | Maximum memory slots (eviction cap) |
| `msa_top_k` | 6 | Top-k slots selected per query token |
| `msa_chunk_size` | 4 | Tokens per slot (mean-pooling compression) |
| `msa_lsh_threshold` | 0 | LSH sublinear routing threshold (0 = disabled) |
| `msa_lsh_hashes` | 8 | LSH random projection hashes |
| `msa_lsh_bits` | 10 | LSH bits per bucket |
| `msa_lsh_probe` | 2 | LSH probe buckets per hash |
| `msa_aux_loss` | true | Load-balance aux loss on MSA router |
| `msa_lb_alpha` | 1.0 | MSA load-balance loss coefficient |
| `msa_adaptive_top_k` | true | Reduce top_k for trivial tokens (MoD skip score) |

### MoE (Mixture of Experts)

| Key | Default | Description |
|-----|---------|-------------|
| `use_moe` | true | Enable MoE SwiGLU |
| `num_experts` | 4 | Number of experts |
| `moe_top_k` | 1 | Top-k experts per token (Switch style) |
| `moe_intermediate_size` | 384 | Intermediate size per expert |
| `moe_alpha` | 0.01 | MoE load-balance aux coefficient |
| `moe_router_temperature` | 1.0 | Router temperature |
| `moe_mod_skip` | true | Mixture-of-Depths skip router |

### NARS

| Key | Default | Description |
|-----|---------|-------------|
| `use_nars` | false | Enable NARS controllers (disabled by default) |

### Regularization

| Key | Default | Description |
|-----|---------|-------------|
| `thinking_noise` | 0.0 | Gaussian noise injected into HRM states (0 = disabled) |
| `hidden_mag_cap` | 0.0 | Runtime magnitude cap on residual stream (0 = disabled) |

### Memory / Performance

| Key | Default | Description |
|-----|---------|-------------|
| `gradient_checkpointing` | true | Activation memory savings via recompute |
| `gc_group_size` | 2 | Group N blocks in one checkpoint call |
| `compile` | true | torch.compile the model |

### Transformer Block

| Key | Default | Description |
|-----|---------|-------------|
| `transformer.hidden_size` | 576 | Must match model.hidden_size |
| `transformer.intermediate_size` | 1536 | SwiGLU intermediate (non-MoE path) |
| `transformer.max_seq_len` | 4096 | RoPE precomputation max |
| `transformer.norm` | rmsnorm | Normalization type |
| `transformer.norm_eps` | 1.0e-6 | RMSNorm epsilon |
| `transformer.num_kv_heads` | 4 | KV heads (GQA) |
| `transformer.num_query_heads` | 8 | Query heads (head_dim 72) |
| `transformer.qk_norm` | true | RMSNorm on Q/K after RoPE |
| `transformer.rope_theta` | 500000.0 | RoPE base frequency |
| `transformer.fuse_qkv` | true | Fused Q,K,V projection |
| `transformer.fuse_gate_up` | true | Fused gate+up in SwiGLU |

### Reasoning Cache (RC)

| Key | Default | Description |
|-----|---------|-------------|
| `rc_enabled` | true | Iterative generate-summarize-cache decoding |
| `rc_iterations` | 3 | RC turns (generate-summarize cycles) |
| `rc_reasoning_budget` | 512 | Max tokens per reasoning trace |
| `rc_summary_budget` | 128 | Max tokens per summary |
| `rc_train_probability` | 0.0 | Probability of RC training step |
| `rc_train_iterations` | 2 | RC iterations during training |

### Precision

| Key | Default | Description |
|-----|---------|-------------|
| `fp16_attention` | true | Cast QKV to fp16 for softmax |
| `fp32_rmsnorm` | true | fp32 variance in RMSNorm |
| `fp32_grad_accum` | false | DISABLED (fused AdamW requires matching dtype) |
| `int8_kv_cache` | true | INT8 KV cache at inference |

---

## Training Section

### Optimizer

| Key | Default | Description |
|-----|---------|-------------|
| `optimizer` | `muon_adamw` | Muon for 2D weights + AdamW for rest |
| `learning_rate` | 0.0004 | AdamW peak LR |
| `muon_lr` | 0.02 | Muon peak LR (canonical Keller Jordan) |
| `muon_momentum` | 0.97 | Muon Nesterov momentum |
| `muon_ns_steps` | 5 | Newton-Schulz iteration count |
| `muon_weight_decay` | 0.5 | Scale-aware wd (||W||_ss ~ 2.0) |
| `weight_decay` | 0.1 | AdamW decoupled weight decay |
| `betas` | [0.9, 0.98] | AdamW betas |
| `eps` | 1.0e-8 | AdamW epsilon |

### Schedule

| Key | Default | Description |
|-----|---------|-------------|
| `schedule` | `wsd` | Warmup-Stable-Decay |
| `warmup_steps` | 1000 | Warmup steps (short due to embedding transfer) |
| `min_lr_ratio` | 0.1 | Cosine decays to min_lr_ratio * lr |
| `cooldown_frac` | 0.10 | WSD decay tail (last 10%) |
| `max_steps` | 146485 | Total training steps (derived from train_tokens) |

### Gradient

| Key | Default | Description |
|-----|---------|-------------|
| `batch_size` | 10 | Micro-batch size (VRAM-optimal for 7GB) |
| `grad_accum_steps` | 2 | Gradient accumulation (effective batch 20) |
| `grad_clip` | 0.0 | DISABLED (structural fixes replace clipping) |
| `gradient_checkpointing` | true | Required for distillation phase on 8GB |
| `magic_norm_max` | 0.0 | Per-blade grad norm clip (DISABLED) |

### Precision

| Key | Default | Description |
|-----|---------|-------------|
| `precision` | `manual_bf16` | Model bf16, no autocast, grads fp32 |

### Composite Loss Weights

| Key | Default | Description |
|-----|---------|-------------|
| `composite_loss.w_ce` | 1.0 | Cross-entropy (main LM loss) |
| `composite_loss.w_aux` | 0.0 | Contrastive auxiliary (DISABLED) |
| `composite_loss.w_iso` | 0.02 | HDIM domain invariant alignment |
| `composite_loss.w_moe` | 0.005 | MoE load-balance |
| `composite_loss.w_msa_lb` | 0.01 | MSA router load-balance |
| `composite_loss.w_gdr_router` | 0.005 | GDR capacity router load-balance |
| `composite_loss.w_quality` | 0.0 | Quality head BCE (disabled) |

### Warmup Start Values

All auxiliary losses ramp from 0 to target:
- `w_aux_start: 0.0`, `w_iso_start: 0.0`, `w_moe_start: 0.0`, `w_msa_lb_start: 0.0`, `w_gdr_router_start: 0.0`

### Checkpointing

| Key | Default | Description |
|-----|---------|-------------|
| `ckpt_interval` | 500 | Checkpoint save interval (steps) |

---

## RL Section

See `configs/rl_rtx3070.yaml` for RL training configuration. Key parameters:

| Key | Default | Description |
|-----|---------|-------------|
| `max_steps` | 10000 | RL training steps |
| `group_size` | 4 | Rollouts per prompt |
| `num_prompts_per_step` | 4 | Prompts per step |
| `max_new_tokens` | 256 | Max generation length |
| `learning_rate` | 1e-5 | RL optimizer LR |
| `mgpo_gamma` | 1.0 | MGPO prompt weight gamma |
| `mgpo_p0` | 0.5 | Target difficulty boundary |

---

## Config Editing Rules

1. **YAML comments with `Read in:` tags are cross-references** to consuming code -- do NOT strip them.
2. **When changing a value, update the comment with the new rationale** -- future tuning depends on this institutional knowledge.
3. **The config is the single source of truth** -- no separate config in code.
4. **Every non-default value has an inline rationale** -- if you add a new key, document why.
