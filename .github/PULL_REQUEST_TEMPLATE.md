## Summary

What does this PR do? One or two sentences.

Closes #<issue-number>

## Type

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `exp` — experiment / ablation
- [ ] `docs` — documentation
- [ ] `refactor` — code restructuring
- [ ] `test` — tests
- [ ] `chore` — tooling / infra

## Changes

- ...
- ...

## Experimental Results (for `exp` PRs)

> Required if this PR changes model architecture or training.

| Metric | Baseline | This PR | Δ |
|--------|----------|---------|---|
| Perplexity | | | |
| GSM8K | | | |
| ARC-Challenge | | | |
| BoolQ | | | |

**Hypothesis:**
**Baseline compared against:**
**Variable under test:**
**Conclusion:**

Config + results saved to: `benchmarks/results/<name>/`

## Checklist

- [ ] Code formatted (`ruff format` / `cargo fmt`)
- [ ] Linters pass (`ruff check` / `cargo clippy -- -D warnings`)
- [ ] Tests added/updated and passing
- [ ] Docs updated if behavior changed
- [ ] No secrets or credentials committed
- [ ] CI green
- [ ] Linked to an issue
