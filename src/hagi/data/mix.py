from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from collections.abc import Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import (
        DataLoader as _TorchDataLoader,
        Dataset as _TorchDataset,
    )
except ImportError:  # pragma: no cover - torch is required for DataLoader use
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass

else:
    DataLoader = cast(Any, _TorchDataLoader)
    Dataset = cast(Any, _TorchDataset)

from hagi.utils import _pad_batch


class WeightedMemmapDataset(Dataset):  # type: ignore[type-arg, misc]
    def __init__(
        self,
        mix: Sequence[tuple[str | Path, float]],
        seq_len: int,
        dtype: str | np.dtype[Any] = "uint16",
        seed: int = 0,
        preload: bool = True,
        min_seq_len: int | None = None,
    ) -> None:
        if not mix:
            raise ValueError("mix must not be empty")
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        # Variable-length training window floor. Defaults to seq_len (fixed
        # length, backward-compatible). When < seq_len, __getitem__ samples a
        # window in [min_seq_len, seq_len]; the collate fn right-pads with
        # ignore_index so shorter samples train loss-free under causal attn.
        self.min_seq_len = int(min_seq_len) if min_seq_len is not None else self.seq_len
        if not (1 <= self.min_seq_len <= self.seq_len):
            raise ValueError(
                f"min_seq_len must be in [1, seq_len={self.seq_len}], "
                f"got {self.min_seq_len}"
            )
        self.dtype = dtype
        self.seed = int(seed)
        self.preload = bool(preload)
        self.paths: list[Path] = []
        self._lengths: list[int] = []
        weights: list[float] = []
        arrays: list[np.ndarray[Any, Any] | None] = []
        for path, weight in mix:
            weight = float(weight)
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError("weight must be positive and finite")
            path = Path(path)
            array = np.memmap(path, dtype=self.dtype, mode="r")
            if self.preload:
                array = np.asarray(array)
            length = len(array)
            if length <= self.seq_len:
                raise ValueError("source length must be greater than seq_len")
            self.paths.append(path)
            self._lengths.append(length)
            arrays.append(array)
            weights.append(weight)
        total = sum(weights)
        self.weights = np.asarray(
            [weight / total for weight in weights], dtype=np.float64
        )
        self._total_length = sum(length - self.seq_len for length in self._lengths)
        self._arrays = arrays

    def _array(self, source_index: int) -> np.ndarray[Any, Any]:
        array = self._arrays[source_index]
        if array is None:
            array = np.memmap(self.paths[source_index], dtype=self.dtype, mode="r")
            if self.preload:
                array = np.asarray(array)
            self._arrays[source_index] = array
        return array

    def __len__(self) -> int:
        return self._total_length

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = np.random.default_rng(self.seed + int(index))
        source_index = rng.choice(len(self.paths), p=self.weights)
        array = self._array(source_index)
        # Variable window: sample length in [min_seq_len, seq_len] so the model
        # sees mixed-context batches (right-padded loss-free by the collate fn).
        win = (
            int(rng.integers(self.min_seq_len, self.seq_len + 1))
            if self.min_seq_len < self.seq_len
            else self.seq_len
        )
        max_start = len(array) - win
        if max_start <= 0:
            raise ValueError("memmap dataset is too small")
        start = rng.integers(0, max_start)
        chunk = np.asarray(array[start : start + win + 1], dtype=np.int64)
        return chunk[:-1], chunk[1:]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if not self.preload:
            state["_arrays"] = [None for _ in self.paths]
        return state


def _mixed_shift_collate(batch: list[tuple[Any, Any]]) -> tuple[Any, Any]:
    # Dataset yields already-split (x=chunk[:-1], y=chunk[1:]) pairs of variable
    # length. Input x pads with the pad_token (id 0 for SmolLM2, a valid embed
    # id); target y pads with ignore_index (-100) so CE skips padded positions.
    # Leak-free under causal right-padding.
    xs = [np.asarray(x, dtype=np.int64) for x, _ in batch]
    ys = [np.asarray(y, dtype=np.int64) for _, y in batch]
    return _pad_batch(xs, pad_value=0), _pad_batch(ys, pad_value=-100)


def get_mixed_memmap_dataloader(
    mix: Sequence[tuple[str | Path, float]],
    batch_size: int,
    seq_len: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    dtype: str | np.dtype[Any] = "uint16",
    seed: int = 0,
    preload: bool = True,
    min_seq_len: int | None = None,
) -> Any:
    if torch is None or DataLoader is None:
        raise ImportError("torch is required for get_mixed_memmap_dataloader")
    dataset = WeightedMemmapDataset(
        mix,
        seq_len=seq_len,
        dtype=dtype,
        seed=seed,
        preload=preload,
        min_seq_len=min_seq_len,
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _mixed_shift_collate,
        "drop_last": True,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = True
    return DataLoader(cast(Any, dataset), **kwargs)
