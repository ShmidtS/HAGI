from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from hagi.data.dataloader import MemmapDataset

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - numpy fallback for non-torch environments
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]


def _as_long_tensor(values: Any) -> Any:
    if torch is None:
        return np.asarray(values, dtype=np.int64)
    return torch.as_tensor(values, dtype=torch.long)


def get_batch_memmap(dataset: Any, batch_size: int, seq_len: int) -> tuple[Any, Any]:
    if len(dataset) <= 0:
        raise ValueError("memmap dataset is too small")
    data_len = len(dataset.data) if hasattr(dataset, "data") else len(dataset) + seq_len
    max_start = data_len - seq_len
    if max_start <= 0:
        raise ValueError("memmap dataset is too small")
    indices = np.random.randint(0, max_start, size=batch_size)
    xs = []
    ys = []
    for index in indices:
        if hasattr(dataset, "data"):
            chunk = np.asarray(dataset.data[index : index + seq_len + 1], dtype=np.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        else:
            x, y = dataset[int(index)]
            xs.append(np.asarray(x, dtype=np.int64)[:seq_len])
            ys.append(np.asarray(y, dtype=np.int64)[:seq_len])
    return _as_long_tensor(np.stack(xs)), _as_long_tensor(np.stack(ys))


def _shift_collate(batch: list[Any]) -> tuple[Any, Any]:
    array = np.stack([np.asarray(item, dtype=np.int64) for item in batch])
    x = array[:, :-1]
    y = array[:, 1:]
    return _as_long_tensor(x), _as_long_tensor(y)


def get_memmap_dataloader(
    memmap_path: str | Path,
    batch_size: int,
    seq_len: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    dtype: str | np.dtype[Any] = np.uint16,
) -> Any:
    if torch is None or DataLoader is None:
        raise ImportError("torch is required for get_memmap_dataloader")
    dataset = MemmapDataset(memmap_path, seq_len=seq_len, dtype=dtype)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _shift_collate,
        "drop_last": True,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def get_batch_synthetic(vocab_size: int, batch_size: int, seq_len: int) -> tuple[Any, Any]:
    if torch is None:
        x = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
        y = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
        return x, y
    x = torch.randint(vocab_size, (batch_size, seq_len), dtype=torch.long)
    y = torch.randint(vocab_size, (batch_size, seq_len), dtype=torch.long)
    return x, y


class BatchLoader:
    def __init__(self, get_batch: Callable[[], tuple[Any, Any]]) -> None:
        self.get_batch = get_batch

    def __iter__(self) -> "BatchLoader":
        return self

    def __next__(self) -> tuple[Any, Any]:
        return self.get_batch()
