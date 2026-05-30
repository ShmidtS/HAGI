"""Benchmark evaluation + intelligence-density metrics.

Two paths:

1. Run benchmarks via lm-eval-harness (registers the HAGI adapter):
       python -m prototype.evaluation.evaluate \
           --ckpt checkpoints/gdr/step-00050000.pt \
           --benchmarks gsm8k,arc_challenge,boolq

2. Compute intelligence-density metrics from raw scores:
       HAGI-IQ  = geomean(reasoning_scores) / model_size_GB
       HAGI-IPP = geomean(reasoning_scores) / active_params_billions
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


def run_lm_eval(ckpt: str, tokenizer: str, benchmarks: list[str], device: str):
    """Run benchmarks through lm-eval-harness."""
    try:
        from lm_eval import simple_evaluate
    except ImportError as e:
        raise SystemExit(f"lm-eval not installed: `pip install lm-eval`. ({e})") from e

    # Importing the wrapper registers the "hagi" model with the harness.
    from prototype.evaluation import lm_eval_wrapper  # noqa: F401

    results = simple_evaluate(
        model="hagi",
        model_args=f"ckpt={ckpt},tokenizer={tokenizer},device={device}",
        tasks=benchmarks,
    )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--benchmarks", default="gsm8k,arc_challenge,boolq")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    benchmarks = args.benchmarks.split(",")
    results = run_lm_eval(args.ckpt, args.tokenizer, benchmarks, args.device)
    print(results.get("results", results))


if __name__ == "__main__":
    main()
