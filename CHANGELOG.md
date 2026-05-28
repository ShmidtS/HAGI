# Changelog

All notable changes to HAGI are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
