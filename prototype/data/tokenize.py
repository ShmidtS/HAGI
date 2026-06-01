"""Tokenize a corpus into flat uint16 .bin shards for MemmapTokenDataset.

Writes exactly the format the loader reads (`prototype/data/dataset.py`): a flat
stream of uint16 token ids per shard, documents separated by the tokenizer's EOS
id. The SmolLM2 vocab (49,152) fits in uint16, so shards are half the size of a
uint32 stream and load directly via memmap — no intermediate format.

Usage:
    python -m prototype.data.tokenize \
        --dataset HuggingFaceFW/fineweb-edu \
        --subset sample-10BT \
        --output data/fineweb-edu \
        --tokenizer HuggingFaceTB/SmolLM2-135M

Add `--limit N` to tokenize only the first N documents (a quick smoke run). Run
once per source (edu text + code + math) and point the loader at the combined
directory, or weight sources via separate runs.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

UINT16_MAX = 65535


def write_shards(
    token_batches: Iterable[np.ndarray],
    output_dir: str | Path,
    shard_size: int = 100_000_000,
    dtype=np.uint16,
) -> list[Path]:
    """Consume arrays of token ids and write flat `.bin` shards of ~shard_size
    tokens each. Returns the shard paths in order. Pure I/O — no tokenizer or
    network — so it is unit-testable on synthetic token streams.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buf: list[np.ndarray] = []
    buf_len = 0
    idx = 0

    def flush():
        nonlocal buf, buf_len, idx
        if buf_len == 0:
            return
        arr = np.concatenate(buf).astype(dtype, copy=False)
        path = out / f"shard_{idx:05d}.bin"
        arr.tofile(path)
        paths.append(path)
        idx += 1
        buf, buf_len = [], 0

    for tb in token_batches:
        a = np.asarray(tb)
        if a.size == 0:
            continue
        buf.append(a)
        buf_len += a.size
        if buf_len >= shard_size:
            flush()
    flush()
    return paths


def _encode_batch(texts: list[str], tokenizer, eos_id: int) -> np.ndarray:
    """Tokenize a batch of documents into one flat int64 array, EOS-separated."""
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    ids: list[int] = []
    for seq in encoded:
        ids.extend(seq)
        ids.append(eos_id)
    return np.asarray(ids, dtype=np.int64)


def _token_batches(
    texts: Iterable[str], tokenizer, eos_id: int, batch_docs: int = 1000
) -> Iterator[np.ndarray]:
    batch: list[str] = []
    for text in texts:
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_docs:
            yield _encode_batch(batch, tokenizer, eos_id)
            batch = []
    if batch:
        yield _encode_batch(batch, tokenizer, eos_id)


def tokenize_corpus(
    dataset: str,
    output: str | Path,
    tokenizer: str = "HuggingFaceTB/SmolLM2-135M",
    subset: str | None = None,
    split: str = "train",
    text_key: str = "text",
    shard_size: int = 100_000_000,
    limit: int | None = None,
) -> list[Path]:
    """Stream `dataset`, tokenize with `tokenizer`, write uint16 `.bin` shards."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    if len(tok) > UINT16_MAX + 1:
        raise ValueError(
            f"tokenizer vocab {len(tok)} does not fit uint16; the loader reads uint16 shards"
        )
    eos = tok.eos_token_id if tok.eos_token_id is not None else len(tok) - 1

    ds = load_dataset(dataset, name=subset, split=split, streaming=True)

    def texts() -> Iterator[str]:
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            yield row[text_key]

    paths = write_shards(_token_batches(texts(), tok, eos), output, shard_size=shard_size)
    total = sum(p.stat().st_size for p in paths) // np.dtype(np.uint16).itemsize
    print(f"wrote {len(paths)} shard(s), ~{total:,} tokens, to {output}")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default=None, help="dataset config/subset, e.g. sample-10BT")
    ap.add_argument("--output", required=True, help="output dir for tokenized .bin shards")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--shard-size", type=int, default=100_000_000, help="tokens per shard")
    ap.add_argument("--limit", type=int, default=None, help="max documents (quick smoke run)")
    args = ap.parse_args()

    try:
        import datasets  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"datasets/transformers not installed. `pip install datasets transformers`. ({e})"
        ) from e

    tokenize_corpus(
        args.dataset, args.output, tokenizer=args.tokenizer, subset=args.subset,
        split=args.split, text_key=args.text_key, shard_size=args.shard_size, limit=args.limit,
    )


if __name__ == "__main__":
    main()
