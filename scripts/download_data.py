from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

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


def _convert_messages_to_dicts(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    messages = row.get("messages")
    if messages is None:
        return None
    if not isinstance(messages, list):
        return None
    return {"messages": messages}


def download_sft_dataset(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install datasets to download SFT data: pip install datasets") from exc

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset, name=args.dataset_config, split=args.split, streaming=False)

    rows: list[dict[str, Any]] = []
    for row in dataset:
        conv = _convert_messages_to_dicts(row)
        if conv is not None:
            rows.append(conv)

    import json

    path = output_dir / "train.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} conversations to {path}")


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
    seen_hashes: set[str] = set()
    skipped = 0
    for row in dataset:
        text = row.get("text", "") if isinstance(row, dict) else ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=8192)
        if len(ids) < args.min_length:
            skipped += 1
            continue
        if len(set(ids)) / max(1, len(ids)) < args.dedup_ratio:
            skipped += 1
            continue
        token_hash = hashlib.sha256(np.asarray(ids, dtype=np.uint16).tobytes()).hexdigest()
        if token_hash in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(token_hash)
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
    if skipped:
        print(f"skipped {skipped} short/duplicate/low-diversity samples")

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
    parser.add_argument("--output", "--output-dir", type=Path, dest="output", default=Path("E:/HAGI/data/fineweb_edu_smollm2"))
    parser.add_argument("--name", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--shard-tokens", type=int, default=10_000_000)
    parser.add_argument("--min-length", type=int, default=50, help="minimum token count per sample")
    parser.add_argument("--dedup-ratio", type=float, default=0.9, help="minimum ratio of unique tokens (diversity filter)")
    parser.add_argument("--dataset", default=None, help="HuggingFace SFT dataset name (e.g. HuggingFaceTB/smoltalk)")
    parser.add_argument("--dataset-config", default="all", help="dataset config/subset name (e.g. 'all' for smoltalk)")
    parser.add_argument("--sft", action="store_true", help="download SFT conversational dataset instead of raw tokens")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sft or args.dataset is not None:
        download_sft_dataset(args)
    else:
        download_and_tokenize(args)


if __name__ == "__main__":
    main()
