# Contributing to HAGI

HAGI is a research project. Contributions are welcome, but the bar is **scientific rigor**: every architectural change must be justified by an ablation or a clear hypothesis.

## Ground Rules

1. **No unvalidated claims.** If you add a mechanism, you must be able to ablate it. "It feels better" is not evidence.
2. **Baseline first.** Every experiment compares against the dense transformer baseline (Model A).
3. **One variable at a time.** Do not stack multiple novel mechanisms in a single experiment. Confounds make results meaningless.
4. **PyTorch for prototyping, Rust for production.** Do not port to Rust until a result is validated.
5. **Reproducibility.** Fixed seeds, logged configs, versioned data.

## How to Contribute

### 1. Find or Open an Issue

- Check [open issues](https://github.com/ShmidtS/HAGI/issues) and [milestones](https://github.com/ShmidtS/HAGI/milestones).
- For new ideas, open a **Research Proposal** issue first to discuss before implementing.

### 2. Set Up Your Environment

```bash
git clone https://github.com/ShmidtS/HAGI.git
cd HAGI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # linting, testing
```

### 3. Branch

```bash
git checkout -b <type>/<short-description>
```

Branch types: `feat/`, `fix/`, `exp/` (experiment), `docs/`, `refactor/`, `test/`.

### 4. Develop

- Follow the code style (see below).
- Add tests for new functionality.
- Update docs if behavior changes.
- For experiments, log results in `benchmarks/results/`.

### 5. Submit a Pull Request

- Fill out the PR template completely.
- Link the related issue.
- For experimental PRs, include benchmark numbers and the ablation comparison.
- Ensure CI passes.

## Code Style

### Python

- Format with `ruff format`.
- Lint with `ruff check`.
- Type hints required for public functions.
- Docstrings for modules and public classes.

```bash
ruff format prototype/
ruff check prototype/
```

### Rust

- Format with `cargo fmt`.
- Lint with `cargo clippy -- -D warnings`.
- No `unwrap()` in library code — use `Result`.

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

### Lean4

- Keep proofs small and composable.
- Comment non-obvious tactic steps.
- `lake build` must pass.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
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

## Review Process

- All PRs require at least one approving review from a code owner (see [CODEOWNERS](.github/CODEOWNERS)).
- Experimental PRs require review of methodology, not just code.
- Maintainers may request additional ablations before merging.

## Questions

Open a [Discussion](https://github.com/ShmidtS/HAGI/discussions) or ask in the relevant issue.
