# Changelog

All notable changes to HAGI are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Training stack** (SmolLM-aligned, composed not hand-rolled):
  - `prototype/training/optim.py` — AdamW + **Muon** hybrid (orthogonalized updates via Newton-Schulz; Muon for core matrices, AdamW for embeddings/norms/gates).
  - `prototype/training/loop.py` — nanoGPT-adapted loop: bf16/fp16 AMP, grad accumulation, cosine+warmup LR, grad clip, checkpointing, eval interval, data-source-agnostic `get_batch`.
  - `prototype/data/` — SmolLM2 tokenizer wrapper, datatrove tokenization script (FineWeb-Edu→.bin shards), memmap dataset loader, toy corpus.
  - `prototype/evaluation/lm_eval_wrapper.py` — HAGI adapter for EleutherAI lm-eval-harness (`loglikelihood`, `generate_until`).
  - `prototype/tests/test_overfit.py` — overfit sanity test (loop correctness for AdamW + Muon). 16 tests pass total.
  - `configs/overfit.yaml`; `docs/TRAINING.md` (workflow + the stack).
- `HAGI.forward(targets=...)` now optionally returns cross-entropy loss (nanoGPT-compatible).

### Changed
- Tokenizer set to **SmolLM2 (~49K vocab)** across configs (was inconsistent Llama-3.2/32K). Right vocab size for 100M scale — embedding ~33% of params, not ~50%.
- `requirements.txt` adds `datatrove` and `lm-eval`.
- Complete repository revamp around **Grade-Decomposed Recurrence (GDR)** architecture.
- `docs/ARCHITECTURE.md` — full GDR specification (Perception → Reasoning → Expression).
- `docs/RESEARCH.md` — literature review with evidence classification (proven / promising / weak / marketing / none).
- `docs/MILESTONES.md` — staged roadmap (Stage 0-6) with gates and stop/pivot conditions.
- Community files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- `.github/` — CODEOWNERS, issue templates (bug / feature / research proposal), PR template, CI workflow.
- `requirements.txt` and `requirements-dev.txt` for the PyTorch prototype.
- `prototype/` scaffold for PyTorch model, training, evaluation.

### Changed
- README fully rewritten to reflect the GDR research direction.
- Architecture pivoted from stacked HRM+HDIM+MSA+MoE+Titans to a single novel mechanism (GDR) with controlled ablation.
- HRM H/L distinction reframed as grade-momentum within a shared block (no architectural duplication).
- Clifford structure moved from bolted-on layer (HDIM) to integral recurrence mechanism (GDR).

### Deferred
- MoE, MSA sparse memory, Titans/TTT memory — removed from core experiment to eliminate confounds. Reintroduced only post-validation.

### Retained
- Rust workspace (`crates/`) — to be refactored for GDR at Stage 5.
- Lean4 formalization (`formalization/`) — to be aligned with implementation at Stage 6.

## [0.1.0] — Prior design (archived direction)

### Added
- Initial Rust workspace scaffold (10 crates).
- Lean4 formalization of core invariants.
- Architecture docs for HRM + HDIM + MSA design.
- cuda-oxide submodule integration.
