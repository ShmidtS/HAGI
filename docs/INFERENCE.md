# HAGI Inference

HAGI provides multiple inference paths: interactive chat REPL, streaming generation, LoRA fine-tuning, online learning, and Reasoning Cache decoding.

For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md). For training, see [TRAINING.md](TRAINING.md).

---

## Quick Start

### Interactive Chat

```bash
python scripts/chat.py --config configs/rtx3070_canonical.yaml --checkpoint checkpoints/rtx3070
```

### Programmatic Generation

```python
from hagi.inference.generate import generate, GenerationConfig

config = GenerationConfig(
    max_new_tokens=128,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
)
result = generate(model, tokenizer, prompt_tokens, config)
```

---

## ChatSession

`hagi.inference.chat.ChatSession` is the main interactive entry point.

### Features

- Multi-turn conversation with history management
- Streaming token output
- LoRA adapters (automatic after N turns)
- Online learning (feedback-driven adaptation)
- Reasoning Cache integration
- MSA persistent slot registry (accumulates across decode steps)
- torch.compile support
- Automatic CUDA cache clearing
- Rollout-based generation (multiple samples + best selection)

### Configuration

```python
session = ChatSession(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    system_prompt="You are HAGI, a helpful assistant.",
    max_context_length=4096,
    compile_model=False,
    lora_rank=8,
    lora_alpha=16,
    auto_learn_after=3,        # auto-apply LoRA after 3 turns
    rc_config=None,            # Reasoning Cache config (optional)
)
```

### Usage

```python
response = session.chat("What is 2+2?")
print(response)

# Stream tokens
for token in session.chat_stream("Tell me a story"):
    print(token, end="", flush=True)
```

Source: `src/hagi/inference/chat.py`

---

## Generation

`hagi.inference.generate` provides the core generation primitives.

### GenerationConfig

Immutable configuration (VibeHarness Config pattern):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 128 | Maximum tokens to generate |
| `temperature` | 1.0 | Sampling temperature |
| `top_k` | 50 | Top-k sampling |
| `top_p` | 0.9 | Nucleus sampling |
| `eos_token_id` | None | End-of-sequence token |
| `stop_sequences` | () | Tuple of token sequences that stop generation |
| `use_cache` | True | Use KV cache |
| `use_static_cache` | False | Use static KV cache (pre-allocated) |
| `compile_model` | False | torch.compile the model |
| `training_mode` | False | Run in training mode (no eval) |
| `early_exit_confidence` | 0.0 | Early exit threshold (0 = disabled) |

### GenerationResult

Structured output (VibeHarness Decision pattern):

| Field | Type | Description |
|-------|------|-------------|
| `token_ids` | list[int] | Generated token IDs |
| `text` | str | Decoded text |
| `finish_reason` | str | "length", "eos", "stop_sequence" |
| `num_tokens` | int | Number of tokens generated |
| `timing` | dict | Per-phase timing (prefill, decode, total) |

### Functions

| Function | Description |
|----------|-------------|
| `generate()` | Standard autoregressive generation |
| `stream_generate()` | Streaming generator (yields tokens) |
| `generate_with_rollouts()` | Multiple rollouts + best selection |

### KV Cache

Two KV cache implementations:

| Type | Description |
|------|-------------|
| Dynamic | Standard PyTorch attention KV cache, grows with generation |
| Static | Pre-allocated cache (INT8 quantized), fixed size |

INT8 KV cache (`int8_kv_cache: true`): quantizes K/V to int8 with per-head fp16 scales. 2x cache memory reduction, enabling longer generation sequences. Quantization is symmetric (abs_max / 127), dequantization is exact.

Source: `src/hagi/inference/generate.py`, `src/hagi/model/kv_cache.py`

---

## CAST Block Generation

When `cast_config.block_size > 1`, each forward pass predicts K=8 tokens via multivector virtual states. This reduces sequential forward passes by 8x during generation.

```
Standard:  128 tokens = 128 forward passes
CAST (K=8): 128 tokens = 16 forward passes
```

Each virtual state is decoded through the shared `final_norm + lm_head`, producing K token predictions per position. The geometric product between adjacent virtual states creates a bivector "area" that enforces cross-token coherence.

Source: `src/hagi/model/cast.py`

---

## Reasoning Cache (RC)

Iterative generate-summarize-cache decoding (arXiv:2602.03773). Replaces standard autoregressive decoding with an iterative loop.

### Algorithm

Per turn t:
1. **Generation**: produce reasoning trace `z_R^(t)` conditioned on (prompt + previous summary + reasoning instruction). Bounded by `H_R` tokens.
2. **Summarization**: produce summary `z_S^(t)` conditioned on (prompt + reasoning trace + previous summary + summary instruction). Bounded by `H_S` tokens (`H_S << H_R`).
3. **Discard** `z_R^(t)`; carry `z_S^(t)` forward as the cache.

The effective reasoning horizon is `T * (H_R + H_S)`, but each generation step operates on bounded context, staying close to the training distribution.

### Configuration

```python
from hagi.inference.reasoning_cache import RCConfig

rc_config = RCConfig(
    iterations=3,              # RC turns
    reasoning_budget=512,      # H_R: max tokens per reasoning trace
    summary_budget=128,        # H_S: max tokens per summary
    use_msa_cache=True,        # Register summaries as MSA slots
)
```

### MSA Integration

When `use_msa_cache=True`, summary hidden states are registered as MSA (Memory Sparse Attention) slots via `external_msa_registry`, leveraging HAGI's sparse attention for cross-iteration memory retrieval.

### Training Integration

RC can be trained during the main training loop:
- `rc_train_probability`: probability of an RC training step
- `rc_train_iterations`: RC iterations during training

Source: `src/hagi/inference/reasoning_cache.py`

---

## LoRA Adapters

`hagi.inference.lora.LoRAAdapter` wraps a linear layer with a low-rank adapter.

### Architecture

```
output = base(x) + scale * (x @ A.T @ B.T)
```

- `A`: `rank x in_features` (random init, scaled 0.01)
- `B`: `out_features x rank` (zero init)
- `scale`: `alpha / rank`
- Base layer parameters are frozen

### Auto-Application

`ChatSession` with `auto_learn_after=3` automatically applies LoRA adapters to all linear layers after 3 conversation turns, enabling in-context adaptation.

### Manual Application

```python
from hagi.inference.lora import apply_lora_to_model

model = apply_lora_to_model(model, rank=8, alpha=16)
```

Source: `src/hagi/inference/lora.py`

---

## Online Learning

`hagi.inference.online.OnlineLearner` provides feedback-driven adaptation during chat.

### FeedbackBuffer

Collects (prompt, response, feedback) tuples. When enough feedback is collected, a mini fine-tuning step is applied to the LoRA adapters.

### Usage

```python
from hagi.inference.online import OnlineLearner, FeedbackBuffer

learner = OnlineLearner(model, lora_params, lr=1e-4)
buffer = FeedbackBuffer(min_size=4)

# After a conversation turn:
buffer.add(prompt, response, feedback="good")
if buffer.ready():
    learner.step(buffer.flush())
```

Source: `src/hagi/inference/online.py`

---

## Inference Optimizations

The model includes several inference-specific optimizations:

| Optimization | Config | Effect |
|-------------|--------|--------|
| INT8 KV cache | `int8_kv_cache: true` | 2x cache memory reduction |
| torch.compile | `compile: true` | Kernel fusion, faster decode |
| CAST block generation | `cast_config.block_size: 8` | 8x fewer forward passes |
| Reasoning Cache | `rc_enabled: true` | Extended reasoning horizon |
| MSA persistent registry | (chat.py) | Cross-step memory accumulation |
| Early exit | `early_exit_confidence` | Stop generation when confident |
| Static KV cache | `use_static_cache` | Pre-allocated cache (no growth) |

### Inference-Only Model

`src/hagi/model/inference_opt.py` provides an inference-optimized model wrapper that strips training-only components (gradient checkpointing, loss computation) for lower memory and faster decode.

Source: `src/hagi/model/inference_opt.py`

---

## MSA at Inference

During training, the MSA slot registry is cleared every forward pass. At inference, `ChatSession` builds a persistent `SlotRegistry(max_slots=msa_slot_count)` that accumulates across decode steps.

This allows the model to build up a memory of the conversation context across turns, retrieving relevant slots via HDIM-invariant routing for each new generation step.

Source: `src/hagi/model/msa.py`, `src/hagi/inference/chat.py`

---

## CLI

### chat.py

```bash
python scripts/chat.py \
    --config configs/rtx3070_canonical.yaml \
    --checkpoint checkpoints/rtx3070 \
    --temperature 0.7 \
    --top-k 50 \
    --top-p 0.9 \
    --max-new-tokens 256
```

### hagi-chat

There is NO `hagi-chat` console script. Only `hagi-train` and `hagi-eval` are in `pyproject.scripts`. Use `python scripts/chat.py` or `python -m hagi.inference.cli`.

Source: `src/hagi/inference/cli.py`
