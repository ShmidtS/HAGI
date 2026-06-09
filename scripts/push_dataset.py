"""Create/refresh an HF *dataset* repo for the tokenized HAGI training shards.

Scans a directory of `shard_*.bin` files (the flat uint16 token-id format that
`prototype/data/dataset.py` reads), writes an auto-statted dataset card (README.md
with the shard table + total token count filled in from the real files), and uploads
the shards. Re-runnable.

    # preview the card only (no repo, no upload)
    python scripts/push_dataset.py --user NAME0x0 --data /content/drive/MyDrive/hagi-data --dry-run

    # create the dataset repo + push card + upload shards (needs a write token in HF_TOKEN)
    python scripts/push_dataset.py --user NAME0x0 --data /content/drive/MyDrive/hagi-data
"""

from __future__ import annotations

# ruff: noqa: E501 — card text is intentionally long string literals
import argparse
import glob
import io
import os

UINT16_BYTES = 2


def _scan(data_dir: str) -> tuple[list[tuple[str, int]], int]:
    """Return [(shard_name, token_count), ...] and the total token count."""
    shards = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    rows, total = [], 0
    for p in shards:
        toks = os.path.getsize(p) // UINT16_BYTES
        total += toks
        rows.append((os.path.basename(p), toks))
    return rows, total


def card(user: str, repo: str, rows: list[tuple[str, int]], total: int,
         tokenizer: str, source: str, subset: str) -> str:
    n = len(rows)
    val = rows[-1][0] if rows else "(none)"
    shard_lines = "\n".join(f"| `{name}` | {toks:,} |" for name, toks in rows) or "| (no shards scanned) | |"
    subset_phrase = f"subset `{subset}`" if subset else ""
    return f"""---
license: odc-by
language:
- en
task_categories:
- text-generation
pretty_name: HAGI tokenized FineWeb-Edu (SmolLM2)
tags:
- hagi
- pretraining
- tokenized
- fineweb-edu
- clifford-algebra
---

# HAGI - Tokenized {source} ({tokenizer.split('/')[-1]} tokenizer)

Pre-tokenized token-id shards used to train the [HAGI](https://github.com/ShmidtS/HAGI)
Stage 0 baseline and the **Grade-Decomposed Recurrence** ablation (models A/B/C/D). A
**tokenized derivative** of [`{source}`](https://huggingface.co/datasets/{source}) {subset_phrase},
published so the exact training corpus loads identically in any environment (Colab,
Kaggle, local) with **no re-tokenization** and no Google Drive access.

## Format - read before using

- **Files:** `shard_NNNNN.bin` - a flat little-endian **`uint16`** stream of token ids,
  no header, documents separated by the tokenizer's EOS id. Read via `numpy.memmap`.
- **Tokenizer:** [`{tokenizer}`](https://huggingface.co/{tokenizer}) - **vocab 49,152**
  (fits uint16). Shards tokenized with **any other tokenizer are incompatible**: the ids
  would index the wrong embeddings (silent garbage), so do not mix sources.
- **Token count of a shard:** filesize / 2.
- **Held-out convention:** the **last shard** (`{val}`) is reserved as validation; train on the rest.

## Contents

- **Shards:** {n}
- **Total tokens:** ~{total:,}

| Shard | Tokens |
|-------|--------|
{shard_lines}

## Load

```python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="{repo}", repo_type="dataset", allow_patterns="*.bin")
# point training at the returned directory:
#   python -m prototype.training.train --config configs/ablation_b.yaml --data <path> ...
```

`MemmapTokenDataset` (`prototype/data/dataset.py`) consumes this directory directly.

## Models trained on this data

- Stage 0 baseline: https://huggingface.co/{user}/hagi-stage0
- Ablation: [`-a`](https://huggingface.co/{user}/hagi-ablation-a) / [`-b`](https://huggingface.co/{user}/hagi-ablation-b) / [`-c`](https://huggingface.co/{user}/hagi-ablation-c) / [`-d`](https://huggingface.co/{user}/hagi-ablation-d)

## License & attribution

Derivative of {source} ({subset or "full"}), released under **ODC-By 1.0** - the upstream
license. Attribute FineWeb-Edu (HuggingFaceFW) on use. Tokenization adds no new content;
it only maps text to {tokenizer} ids.
"""


def main():
    ap = argparse.ArgumentParser(description="Push tokenized shards + a data card to an HF dataset repo.")
    ap.add_argument("--user", required=True, help="HF username")
    ap.add_argument("--data", required=True, help="directory of shard_*.bin")
    ap.add_argument("--repo-name", default="hagi-fineweb-edu-smollm2", help="dataset repo name under --user")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--source", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    ap.add_argument("--no-data", action="store_true", help="push only the card, skip uploading shards")
    ap.add_argument("--dry-run", action="store_true", help="print the card; do not create or upload")
    args = ap.parse_args()

    repo = f"{args.user}/{args.repo_name}"
    rows, total = _scan(args.data)
    if not rows and not args.dry_run:
        raise SystemExit(f"no *.bin shards in {args.data}")
    text = card(args.user, repo, rows, total, args.tokenizer, args.source, args.subset)

    if args.dry_run:
        print(text)
        print(f"\n[dry-run] {len(rows)} shards, ~{total:,} tokens -> would push to {repo}")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_file(path_or_fileobj=io.BytesIO(text.encode("utf-8")),
                    path_in_repo="README.md", repo_id=repo, repo_type="dataset")
    print(f"pushed data card -> https://huggingface.co/datasets/{repo}")
    if not args.no_data:
        print(f"uploading {len(rows)} shards (~{total:,} tokens); binary/LFS, may take a while...")
        api.upload_folder(folder_path=args.data, repo_id=repo, repo_type="dataset",
                          allow_patterns="*.bin")
        print(f"uploaded shards -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
