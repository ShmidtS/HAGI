# HAGI — Hypercomplex Artificial General Intelligence

HAGI unites a hierarchical recurrent language model architecture with a geometric structural layer of hypercomplex invariants (HDIM) and a Python-first training/inference stack with CUDA support.

**Branches:**
- `main` — original Rust architecture with Lean4 formalization
- `test` — Rust end-to-end training/evaluation stack
- `test-python` — **Python migration** (active development) with PyTorch, GDR, NARS, and GPU training
- `experimental` — PyTorch GDR prototype and research docs

---

## Python Stack (test-python branch)

### Quick Start

```bash
# Clone and checkout branch
git clone https://github.com/ShmidtS/HAGI.git
cd HAGI
git checkout test-python

# Install dependencies
pip install -e .

# Download training data (FineWeb-Edu subset)
python scripts/download_data.py --subset 10M --output data/fineweb_10M

# Train on RTX 3070 (8GB VRAM)
python scripts/train_rtx3070.py --device cuda --max-steps 50000 --data-dir data/fineweb_10M --seq-len 512

# Chat with trained model
python scripts/chat_rtx3070.py --checkpoint checkpoints/rtx3070/step-XXXXXXXX.pt
```

### Architecture

| Component | Path | Description |
|---|---|---|
| Core Types | `src/hagi/core/` | Tensor specs, shapes, layouts, algebra helpers |
| NARS Core | `src/hagi/nars/` | Term, Truth, Budget, Sentence, Task, Concept, Bag |
| NARS Adapters | `src/hagi/hrm/`, `hdim/`, `msa/` | Control, reasoning, sparse attention adapters |
| Model | `src/hagi/model/` | Clifford algebra, GDR, Transformer, HAGI (4 ablations) |
| Training | `src/hagi/train/` | Loop, optimizer (AdamW/Muon), checkpointing, config |
| Inference | `src/hagi/inference/` | Generation with KV-cache, chat session, streaming |
| Data | `src/hagi/data/` | Tokenizer (SmolLM2), memmap dataset, batching |
| Eval | `src/hagi/eval/` | Golden tests, reports, lm-eval adapter |
| Lean Bridge | `src/hagi/lean/` | Subprocess verification wrapper |

### Model Configurations

Four ablation variants via `HAGIConfig`:
- **A (baseline)**: `use_loop=False, use_gdr=False` — dense transformer
- **B (loop)**: `use_loop=True, use_gdr=False` — recurrent transformer
- **C (HDIM)**: `use_loop=False, use_gdr=True` — Clifford projection without loop
- **D (GDR)**: `use_loop=True, use_gdr=True` — full Grade-Decomposed Recurrence

### RTX 3070 Optimized Training

Config: `configs/rtx3070.yaml`
- **GPU**: RTX 3070 Laptop 8GB
- **Precision**: fp16 (Ampere-optimal)
- **Model**: 53M parameters, hidden_size=512, 3 layers per stage
- **Batch**: 1 physical + 16 gradient accumulation = effective batch 16
- **VRAM**: ~2GB allocated during training (room for larger models)
- **Loss**: Cross-entropy + optional GDR auxiliary + isomorphic consistency

### Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

55+ tests covering: core types, NARS truth/budget, Clifford algebra, model variants, checkpoints, training loop, data pipeline, Lean bridge.

---

## Rust Stack (main/test branches)

See original documentation in `docs/` for:
- HRM hierarchical recurrent model
- HDIM hypercomplex invariants
- MoE + Titans/TTT memory
- CUDA kernels via cuda-oxide
- Lean4 formalization (`formalization/`)

---

## License

[Apache License 2.0](LICENSE).
