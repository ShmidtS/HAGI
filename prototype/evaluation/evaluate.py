"""Benchmark evaluation harness.

Usage:
    python -m prototype.evaluation.evaluate --model checkpoints/gdr_final.pt \
        --benchmarks gsm8k,arc_challenge,boolq

Computes the ablation comparison and the intelligence-density metrics:

    HAGI-IQ  = geomean(reasoning_scores) / model_size_GB
    HAGI-IPP = geomean(reasoning_scores) / active_params_billions

TODO: wire in lm-eval-harness or custom benchmark runners. The metric helpers
below are ready to use once raw scores are produced.
"""

from __future__ import annotations

import argparse
import math


def geomean(values: list[float]) -> float:
    values = [max(v, 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in values) / len(values))


def hagi_iq(reasoning_scores: dict[str, float], model_size_gb: float) -> float:
    return geomean(list(reasoning_scores.values())) / max(model_size_gb, 1e-9)


def hagi_ipp(reasoning_scores: dict[str, float], active_params_b: float) -> float:
    return geomean(list(reasoning_scores.values())) / max(active_params_b, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmarks", default="gsm8k,arc_challenge,boolq")
    args = ap.parse_args()

    benchmarks = args.benchmarks.split(",")
    # TODO: load checkpoint, run each benchmark, collect scores.
    raise SystemExit(
        f"Benchmark runners not yet implemented. Requested: {benchmarks}.\n"
        "Integrate lm-eval-harness or custom runners, then use hagi_iq / hagi_ipp."
    )


if __name__ == "__main__":
    main()
