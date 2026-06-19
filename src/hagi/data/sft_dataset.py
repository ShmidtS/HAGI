"""SFT (Supervised Fine-Tuning) dataset loader for conversational data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hagi.utils import _as_long_tensor

Dataset: Any
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover
    torch: Any = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc,assignment]
else:
    Dataset: Any = _TorchDataset  # type: ignore[assignment]


@dataclass
class SFTExample:
    input_ids: list[int]
    labels: list[int]


def _encode_conversation(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_seq_len: int,
    pad_token_id: int = 0,
) -> SFTExample:
    """Tokenize a conversation and build assistant-only loss mask."""
    has_chat_template = getattr(
        tokenizer, "chat_template", None
    ) is not None and hasattr(tokenizer, "apply_chat_template")

    def _format_turn(msg: dict[str, str]) -> str:
        if has_chat_template:
            text = tokenizer.apply_chat_template(
                [{"role": msg["role"], "content": msg["content"]}],
                tokenize=False,
                add_generation_prompt=False,
            )
            return str(text)
        role = msg.get("role", "")
        content = msg.get("content", "")
        return f"<{role}>{content}</{role}>"

    full_text_parts: list[str] = []
    for msg in messages:
        full_text_parts.append(_format_turn(msg))
    full_text = "".join(full_text_parts)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Determine assistant token boundaries by encoding each turn individually
    assistant_ranges: list[tuple[int, int]] = []
    cursor = 0
    for msg in messages:
        if msg.get("role") not in {"user", "assistant", "system"}:
            continue
        turn_text = _format_turn(msg)
        turn_ids = tokenizer.encode(turn_text, add_special_tokens=False)
        turn_len = len(turn_ids)
        end = cursor + turn_len
        if msg.get("role") == "assistant":
            assistant_ranges.append((cursor, end))
        cursor = end

    # Truncate
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]

    labels = [-100] * len(full_ids)
    for start, end in assistant_ranges:
        start = min(start, len(full_ids))
        end = min(end, len(full_ids))
        for idx in range(start, end):
            labels[idx] = full_ids[idx]

    # Pad
    pad_len = max_seq_len - len(full_ids)
    if pad_len > 0:
        full_ids = full_ids + [pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

    return SFTExample(input_ids=full_ids, labels=labels)


class SFTDataset(Dataset):  # type: ignore[type-arg, misc]
    """PyTorch Dataset wrapper for SFT conversational data."""

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        max_seq_len: int = 512,
        messages_key: str = "messages",
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.messages_key = messages_key
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[index]
        messages = row[self.messages_key]
        if not isinstance(messages, list):
            raise TypeError(f"expected list of messages, got {type(messages).__name__}")
        example = _encode_conversation(
            messages,
            self.tokenizer,
            self.max_seq_len,
            self.pad_token_id,
        )
        if torch is not None:
            return {
                "input_ids": torch.tensor(example.input_ids, dtype=torch.long),
                "labels": torch.tensor(example.labels, dtype=torch.long),
            }
        return {
            "input_ids": np.asarray(example.input_ids, dtype=np.int64),
            "labels": np.asarray(example.labels, dtype=np.int64),
        }


def _sft_collate(batch: list[dict[str, Any]]) -> tuple[Any, Any]:
    input_ids = np.stack(
        [np.asarray(item["input_ids"], dtype=np.int64) for item in batch]
    )
    labels = np.stack([np.asarray(item["labels"], dtype=np.int64) for item in batch])
    return _as_long_tensor(input_ids), _as_long_tensor(labels)


def get_sft_dataloader(
    dataset_name: str,
    tokenizer: Any,
    max_seq_len: int = 512,
    batch_size: int = 2,
    num_workers: int = 2,
    pin_memory: bool = True,
    split: str = "train",
    streaming: bool = False,
    local_path: str | Path | None = None,
    messages_key: str = "messages",
) -> Any:
    """Build a DataLoader for SFT conversational datasets."""
    if torch is None:
        raise ImportError("torch is required for get_sft_dataloader")
    from torch.utils.data import DataLoader

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "install datasets for SFT data loading: pip install datasets"
        ) from exc

    if local_path is not None and Path(local_path).exists():
        suffix = Path(local_path).suffix.lower()
        if suffix == ".jsonl":
            hf_dataset = load_dataset("json", data_files=str(local_path), split=split)
        else:
            hf_dataset = load_dataset(
                "parquet", data_files=str(local_path), split=split
            )
    else:
        hf_dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    dataset = SFTDataset(
        hf_dataset,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        messages_key=messages_key,
    )

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": not streaming,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _sft_collate,
        "drop_last": True,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = True

    return DataLoader(dataset, **kwargs)
