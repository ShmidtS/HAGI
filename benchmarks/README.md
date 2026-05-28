# Benchmarks

Benchmark scripts and experiment results.

## Results Layout

Each experiment writes to `results/<experiment-name>/`:

```
results/<experiment-name>/
├── config.yaml       # exact config used
├── metrics.json      # all benchmark scores + perplexity
├── ablation.md       # comparison table vs baseline
└── notes.md          # hypothesis, conclusion
```

## Benchmark Suite

| Benchmark | Category | Tests |
|-----------|----------|-------|
| GSM8K | Math | Multi-step arithmetic — primary recurrence signal |
| ARC-Challenge | Science | Relational reasoning — primary GDR signal |
| BoolQ | Logic | Directional inference |
| WinoGrande | Commonsense | Coreference / causal structure |
| HellaSwag | Commonsense | General quality (must not regress) |
| HumanEval | Code | Structured generation |
| MMLU (5-shot) | Knowledge | Breadth reference |
| Perplexity | LM | Held-out language modeling |

## Intelligence-Density Metrics

- **HAGI-IQ** = geomean(reasoning_scores) / model_size_GB
- **HAGI-IPP** = geomean(reasoning_scores) / active_params_B

See `prototype/evaluation/evaluate.py` for the implementations.

## Ablation Protocol

The decisive comparison is **Model B vs Model D** — both use recurrence, both
have ~the same parameter count; the only difference is grade decomposition.

- D > B on ≥2 reasoning benchmarks (≥2% absolute) → grade decomposition works.
- D ≈ B → neutral; fall back to recurrence + training recipe.
- D < B → harmful; investigate momentum coefficients / grade partition.
