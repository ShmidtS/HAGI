from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from hagi.data.tokenizer import SMOLLM2_TOKENIZER, TokenizerWrapper

DATASET_NAME = "HuggingFaceFW/fineweb-edu"

DATASET_PRESETS = {
    "edu": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "license_note": "Use streaming/subsampling for RTX 3070 experiments.",
    },
    "cosmopedia": {
        "dataset": "HuggingFaceTB/smollm-corpus",
        "name": "cosmopedia-v2",
        "split": "train",
        "license_note": "Synthetic educational corpus; validate locally before mixing.",
    },
    "smoltalk": {
        "dataset": "HuggingFaceTB/smoltalk",
        "name": "all",
        "split": "train",
        "license_note": "Conversational SFT corpus; validate dataset card before redistribution.",
    },
    "python_instruct": {
        "dataset": "iamtarun/python_code_instructions_18k_alpaca",
        "split": "train",
        "license_note": "Open Python instruction/code corpus used as an unauthenticated code-data fallback.",
    },
    "smollm_corpus": {
        "dataset": "HuggingFaceTB/smollm-corpus",
        "subsets": ["cosmopedia-v2", "fineweb-edu-dedup", "python-edu"],
        "license_note": "See Hugging Face dataset card before redistribution.",
    },
    "fineweb_edu_10bt": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "license_note": "Use streaming/subsampling for RTX 3070 experiments.",
    },
    "cosmopedia_v2": {
        "dataset": "HuggingFaceTB/cosmopedia-v2",
        "license_note": "Synthetic educational corpus; validate locally before mixing.",
    },
    "tinystories": {
        "dataset": "roneneldan/TinyStories",
        "split": "train",
        "license_note": "CDLA-Sharing-1.0; synthetic short stories for simple-language modeling.",
    },
    "wikitext": {
        "dataset": "Salesforce/wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "train",
        "license_note": "CC-BY-SA-4.0; Wikipedia-derived, high-quality factual text.",
    },
    "openwebtext": {
        "dataset": "Skylion007/openwebtext",
        "split": "train",
        "license_note": "CC-derived web corpus; GPT-2 training data, broad-coverage noise.",
    },
    "tinycodes": {
        "dataset": "nampdn-ai/tiny-codes",
        "split": "train",
        "license_note": "Apache-2.0; small code instruction snippets for code reasoning.",
    },
}

# arch_decision §Data adapted for unauthenticated download: 70% FineWeb-Edu / 15% Cosmopedia v2 / 10% SmolTalk / 5% Python instruction code.
DEFAULT_MIX = {
    "edu": 0.70,
    "cosmopedia": 0.15,
    "smoltalk": 0.10,
    "python_instruct": 0.05,
}

# arch_decision §Data v2 scale-up: 50M-token RTX 3070 friendly mix spanning
# web (edu+openwebtext), synthetic textbook (cosmopedia), wiki (wikitext),
# instruction (smoltalk+python_instruct+tinycodes), and narrative (tinystories).
MIX_PRESETS: dict[str, dict[str, float]] = {
    "edu70_cosmo15_chat10_py5": dict(DEFAULT_MIX),
    "edu70_cosmo15_chat10_code5": dict(DEFAULT_MIX),
    "default": dict(DEFAULT_MIX),
    "v2_50m": {
        "edu": 0.5102,
        "cosmopedia": 0.1531,
        "wikitext": 0.1020,
        "smoltalk": 0.0816,
        "tinystories": 0.0612,
        "python_instruct": 0.0612,
        "openwebtext": 0.0307,
    },
}


def parse_mix(value: str | None) -> dict[str, float]:
    """Parse --mix flag into a {source: ratio} dict.

    Accepts a preset name (looked up in MIX_PRESETS) or a comma-separated
    ``name:ratio`` list. Ratios are normalized to sum to 1.0.
    """
    if not value:
        return dict(DEFAULT_MIX)
    if value in MIX_PRESETS:
        return dict(MIX_PRESETS[value])
    out: dict[str, float] = {}
    for part in value.split(","):
        if ":" not in part:
            raise ValueError(f"invalid --mix segment {part!r} (expected name:ratio)")
        name, ratio = part.split(":", 1)
        out[name.strip()] = float(ratio)
    total = sum(out.values()) or 1.0
    return {name: ratio / total for name, ratio in out.items()}


def write_mix_manifest(
    output_dir: Path,
    mix: dict[str, float],
    packing: str = "bfd",
    token_count: int | None = None,
) -> Path:
    """Write data/mix.json manifest (no actual download, code path only)."""
    import json
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "mix.json"
    payload = {
        "version": 1,
        "packing": packing,
        "sources": [{"name": name, "ratio": ratio} for name, ratio in mix.items()],
        "mix": mix,
        "token_count": token_count,
        "presets": {name: DATASET_PRESETS.get(name, {}) for name in mix},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


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


def materialize_token_bins(output_dir: Path, tokens_by_source: dict[str, list[int]]) -> dict[str, Path]:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    manifest: dict[str, Any] = {"version": 1, "sources": {}}
    for name, tokens in tokens_by_source.items():
        path = output_dir / f"{name}.bin"
        array = np.asarray(tokens, dtype=np.uint16)
        memmap = np.memmap(path, dtype=np.uint16, mode="w+", shape=array.shape)
        memmap[:] = array[:]
        memmap.flush()
        paths[name] = path
        manifest["sources"][name] = {"path": path.name, "tokens": int(array.size), "dtype": "uint16"}
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def _row_text(source: str, row: dict[str, Any]) -> str:
    if source == "smoltalk":
        messages = row.get("messages", [])
        return "\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))
    if source == "python_instruct":
        parts = [row.get("instruction", ""), row.get("input", ""), row.get("output", "")]
        return "\n".join(str(part) for part in parts if part)
    if source == "tinycodes":
        for key in ("text", "code", "content"):
            value = row.get(key)
            if value:
                return str(value)
        return ""
    # edu, cosmopedia, tinystories, wikitext, openwebtext: all expose "text".
    return str(row.get("text", ""))


def _dataset_spec_for_source(source: str) -> tuple[str, str | list[str] | None, str]:
    if source == "python_instruct":
        return "iamtarun/python_code_instructions_18k_alpaca", None, "train"
    preset = DATASET_PRESETS[source]
    return str(preset["dataset"]), preset.get("name"), str(preset.get("split", "train"))  # type: ignore


def download_mixed_token_bins(args: argparse.Namespace) -> dict[str, Path]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install datasets to download mixed data: pip install datasets") from exc

    tokenizer = TokenizerWrapper.smollm2(SMOLLM2_TOKENIZER, use_fast=True)
    target_tokens = parse_token_count(args.subset)
    tokens_by_source: dict[str, list[int]] = {}
    paths: dict[str, Path] = {}
    skip_existing = bool(getattr(args, "skip_existing", False))
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, ratio in args.mix_ratios.items():
        target_path = output_dir / f"{source}.bin"
        if skip_existing and target_path.exists():
            size = target_path.stat().st_size
            if size > 0 and size % 2 == 0:
                existing = np.memmap(target_path, dtype=np.uint16, mode="r")
                existing_count = int(existing.shape[0])
                del existing
                if existing_count > 0:
                    paths[source] = target_path
                    print(f"source={source} skip_existing tokens={existing_count}")
                else:
                    print(f"source={source} existing bin is empty (tokens=0); will re-download")
            else:
                print(f"source={source} existing bin size={size}B (invalid); will re-download")
            continue
        source_target = max(args.min_source_tokens, int(target_tokens * ratio))
        dataset_name, dataset_config, split = _dataset_spec_for_source(source)
        load_kwargs: dict[str, Any] = {"split": split, "streaming": True}
        if dataset_config:
            load_kwargs["name"] = dataset_config
        try:
            dataset = load_dataset(dataset_name, **load_kwargs)
            tokens: list[int] = []
            skipped = 0
            for row in dataset:
                text = _row_text(source, row if isinstance(row, dict) else {})
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=8192)
                if len(ids) < args.min_length:
                    skipped += 1
                    continue
                if tokenizer.eos_token_id is not None:
                    ids.append(int(tokenizer.eos_token_id))
                remaining = source_target - len(tokens)
                tokens.extend(ids[:remaining])
                if len(tokens) >= source_target:
                    break
        except Exception as exc:  # HF streaming is fragile on long iterations; skip & continue.
            print(f"source={source} dataset={dataset_name} FAILED error={exc!r}")
            continue
        if not tokens:
            print(f"source={source} dataset={dataset_name} empty (no rows yielded)")
            continue
        tokens_by_source[source] = tokens
        # Flush after each source so a crash later does not lose progress.
        materialize_token_bins(output_dir, {source: tokens})
        paths[source] = output_dir / f"{source}.bin"
        print(f"source={source} dataset={dataset_name} tokens={len(tokens)} skipped={skipped}")
    return paths


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
                "dtype=uint16",
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
    parser.add_argument("--materialize-mix", action="store_true", help="download each mix source into source-named .bin files")
    parser.add_argument("--skip-existing", action="store_true", help="skip sources whose <source>.bin already exists in --output")
    parser.add_argument("--min-source-tokens", type=int, default=1024, help="minimum tokens per materialized mix source")
    parser.add_argument(
        "--mix",
        default="edu70_cosmo15_chat10_code5",
        help="data mix preset or comma-separated name:ratio list",
    )
    parser.add_argument(
        "--packing",
        choices=("bfd", "random"),
        default="bfd",
        help="sequence packing strategy (bfd=best-fit-decreasing on EOS, random=legacy memmap)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.mix_ratios = parse_mix(args.mix)
    if args.packing == "bfd":
        manifest = write_mix_manifest(
            args.output,
            args.mix_ratios,
            packing="bfd",
            token_count=parse_token_count(args.subset),
        )
        print(f"wrote mix manifest {manifest} (sources={list(args.mix_ratios)})")
    if args.materialize_mix:
        paths = download_mixed_token_bins(args)
        print(f"wrote materialized mix bins {paths}")
    elif args.sft or args.dataset is not None:
        download_sft_dataset(args)
    else:
        download_and_tokenize(args)


if __name__ == "__main__":
    main()
