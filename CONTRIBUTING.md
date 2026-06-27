# Contributing to HAGI

HAGI is a research project. Contributions are welcome, but the bar is **scientific rigor**: every architectural change must be justified by an ablation or a clear hypothesis.

## Ground Rules

1. **No unvalidated claims.** If you add a mechanism, you must be able to ablate it. "It feels better" is not evidence.
2. **Baseline first.** Every experiment compares against the dense transformer baseline (Model A).
3. **One variable at a time.** Do not stack multiple novel mechanisms in a single experiment. Confounds make results meaningless.
4. **PyTorch for prototyping.** No Rust/CUDA port until architecture is validated.
5. **Reproducibility.** Fixed seeds, logged configs, versioned data.

## How to Contribute

### 1. Find or Open an Issue

- Check open issues and milestones on GitHub.
- For new ideas, open a **Research Proposal** issue first to discuss before implementing.

### 2. Set Up Your Environment

```bash
git clone https://github.com/ShmidtS/HAGI.git
cd HAGI
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 3. Branch

```bash
git checkout -b <type>/<short-description>
```

Branch types: `feat/`, `fix/`, `exp/` (experiment), `docs/`, `refactor/`, `test/`.

### 4. Develop

- Follow the code style (see below).
- For experiments, log results in `benchmarks/results/`.
- Update docs if behavior changes.

### 5. Submit a Pull Request

- Fill out the PR template completely.
- Link the related issue.
- For experimental PRs, include benchmark numbers and the ablation comparison.
- Ensure `ruff check .` passes.

## Code Style

### Python

- Format with `ruff format`.
- Lint with `ruff check`.
- Type hints required for public functions.
- Docstrings for modules and public classes.

```bash
ruff check .
basedpyright src/hagi
```

### Commit Messages



```
<type>(<scope>): <subject>

<body>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `exp`, `chore`, `perf`.

Examples:
```
feat(model): add grade decomposition to reasoning core
exp(gdr): ablation of momentum coefficients per grade
fix(clifford): correct sign in Cl(3,0,0) product table
```

## Experiment Protocol

When contributing an experiment:

1. State the hypothesis in the PR description.
2. Specify which baseline you compare against.
3. Use identical training config except the variable under test.
4. Report: perplexity, all benchmark scores, training stability.
5. Save the config and a results summary to `benchmarks/results/<experiment-name>/`.

## Config Editing

`configs/rtx3070_canonical.yaml` is the single source of truth. When changing a value:

1. **Update the inline comment** with the new rationale.
2. **Do NOT strip `Read in:` tags** -- they are cross-references to consuming code.
3. **Document why** the new value is better than the old one.

## Architecture Awareness

HAGI is **not** a vanilla transformer. Before making changes:

- Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the forward pass pipeline.
- Read [AGENTS.md](AGENTS.md) for critical runtime invariants.
- Understand that `use_fused_ce: true` means `logits` is `None` in training forward.
- Understand that `precision: manual_bf16` is NOT bf16-autocast.

## Review Process

- All PRs require at least one approving review.
- Experimental PRs require review of methodology, not just code.
- Maintainers may request additional ablations before merging.

## Questions

Open a Discussion on GitHub or ask in the relevant issue.
