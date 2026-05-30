"""Tokenize a corpus into .bin shards using datatrove (the SmolLM data stack).

This produces the memmap shards consumed by dataset.MemmapTokenDataset. It is a
script, not run automatically — invoke explicitly once data + tokenizer are set.

Usage:
    python -m prototype.data.tokenize \
        --dataset HuggingFaceFW/fineweb-edu \
        --subset sample-10BT \
        --output data/fineweb-edu \
        --tokenizer HuggingFaceTB/SmolLM2-135M

datatrove handles streaming, dedup-aware reading, multi-worker tokenization, and
shard writing. We then expose its output as flat token .bin shards. For the HAGI
data mix (edu text + code + math), run this once per source and point the loader
at the combined directory, or weight sources via separate runs.

Reference: HuggingFace datatrove (https://github.com/huggingface/datatrove) and
the Smol Training Playbook for the recommended FineWeb-Edu + code + math mix.
"""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default=None, help="dataset config/subset, e.g. sample-10BT")
    ap.add_argument("--output", required=True, help="output dir for tokenized shards")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    try:
        from datatrove.executor import LocalPipelineExecutor
        from datatrove.pipeline.readers import HuggingFaceDatasetReader
        from datatrove.pipeline.tokens import DocumentTokenizer
    except ImportError as e:
        raise SystemExit(
            "datatrove not installed. `pip install datatrove`. "
            f"(import error: {e})"
        ) from e

    reader = HuggingFaceDatasetReader(
        dataset=args.dataset,
        dataset_options={"name": args.subset} if args.subset else {},
        text_key="text",
    )
    tokenizer = DocumentTokenizer(
        output_folder=args.output,
        tokenizer_name_or_path=args.tokenizer,
        shuffle=True,
    )
    executor = LocalPipelineExecutor(
        pipeline=[reader, tokenizer],
        tasks=args.workers,
    )
    executor.run()
    print(f"tokenized shards written to {args.output}")


if __name__ == "__main__":
    main()
