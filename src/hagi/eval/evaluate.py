"""Benchmark evaluation + intelligence-density metrics.

Two paths:

1. Run benchmarks via lm-eval-harness (registers the HAGI adapter):
       python -m hagi.eval.evaluate \
           --ckpt checkpoints/gdr/step-00050000.pt \
           --benchmarks gsm8k,arc_challenge,boolq

2. Compute intelligence-density metrics from raw scores:
       HAGI-IQ  = geomean(reasoning_scores) / model_size_GB
       HAGI-IPP = geomean(reasoning_scores) / active_params_billions
"""

from __future__ import annotations

import argparse


def run_lm_eval(ckpt: str, tokenizer: str, benchmarks: list[str], device: str):
    """Run benchmarks through lm-eval-harness."""
    try:
        from lm_eval import simple_evaluate  # type: ignore[reportMissingImports]
    except ImportError as e:
        raise SystemExit(f"lm-eval not installed: `pip install lm-eval`. ({e})") from e

    # Importing the wrapper registers the "hagi" model with the harness.
    from hagi.eval import lm_eval_wrapper  # noqa: F401

    return simple_evaluate(
        model="hagi",
        model_args=f"ckpt={ckpt},tokenizer={tokenizer},device={device}",
        tasks=benchmarks,
    )


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
