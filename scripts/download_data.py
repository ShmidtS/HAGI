from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hagi.data.tokenizer import SMOLLM2_TOKENIZER, TokenizerWrapper

DATASET_NAME = "HuggingFaceFW/fineweb-edu"


def parse_token_count(value: str) -> int:
    text = value.strip().lower().replace("_", "")
    multiplier = 1
    if text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    return int(float(text) * multiplier)


def flush_shard(tokens: list[int], output_dir: Path, shard_idx: int) -> Path:
    path = output_dir / f"fineweb_edu_{shard_idx:05d}.bin"
    array = np.asarray(tokens, dtype=np.uint16)
    memmap = np.memmap(path, dtype=np.uint16, mode="w+", shape=array.shape)
    memmap[:] = array[:]
    memmap.flush()
    return path


def download_and_tokenize(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install datasets to download FineWeb-Edu: pip install datasets") from exc

    target_tokens = parse_token_count(args.subset)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = TokenizerWrapper.smollm2(SMOLLM2_TOKENIZER, use_fast=True)
    dataset = load_dataset(DATASET_NAME, name=args.name, split=args.split, streaming=True)

    shard_tokens: list[int] = []
    total_tokens = 0
    shard_idx = 0
    written: list[Path] = []
    for row in dataset:
        text = row.get("text", "") if isinstance(row, dict) else ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            ids.append(int(tokenizer.eos_token_id))
        remaining = target_tokens - total_tokens
        if remaining <= 0:
            break
        ids = ids[:remaining]
        shard_tokens.extend(ids)
        total_tokens += len(ids)
        while len(shard_tokens) >= args.shard_tokens:
            written.append(flush_shard(shard_tokens[: args.shard_tokens], output_dir, shard_idx))
            shard_tokens = shard_tokens[args.shard_tokens :]
            shard_idx += 1
        if total_tokens >= target_tokens:
            break

    if shard_tokens:
        written.append(flush_shard(shard_tokens, output_dir, shard_idx))

    meta = output_dir / "metadata.txt"
    meta.write_text(
        "\n".join(
            [
                f"dataset={DATASET_NAME}",
                f"name={args.name}",
                f"split={args.split}",
                f"tokenizer={SMOLLM2_TOKENIZER}",
                f"tokens={total_tokens}",
                f"dtype=uint16",
                *[f"shard={path.name}" for path in written],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {total_tokens} tokens to {output_dir} in {len(written)} shard(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and tokenize a FineWeb-Edu subset for HAGI.")
    parser.add_argument("--subset", default="10M", help="target token count, e.g. 10M or 100M")
    parser.add_argument("--output", type=Path, default=Path("E:/HAGI/data/fineweb_edu_smollm2"))
    parser.add_argument("--name", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--shard-tokens", type=int, default=10_000_000)
    return parser.parse_args()


def main() -> None:
    download_and_tokenize(parse_args())


if __name__ == "__main__":
    main()
