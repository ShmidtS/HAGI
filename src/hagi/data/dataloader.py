from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    Dataset = cast(Any, _TorchDataset)
except ImportError:  # pragma: no cover - torch is required for DataLoader use
    torch: Any = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass


class MemmapDataset(Dataset):  # type: ignore[type-arg, misc]
    def __init__(
        self,
        path: str | Path,
        seq_len: int | None = None,
        dtype: str | np.dtype[Any] = "uint16",
        mode: str = "r",
        block_size: int | None = None,
        preload: bool = True,
        min_seq_len: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.seq_len = int(
            seq_len
            if seq_len is not None
            else block_size if block_size is not None else 0
        )
        # When min_seq_len is set (< seq_len), __getitem__ samples a random
        # window length in [min_seq_len, seq_len] per item. The collate fn pads
        # to the batch max with ignore_index so shorter samples don't waste
        # compute. Defaults to seq_len (fixed length, backward-compatible).
        self.min_seq_len = int(min_seq_len) if min_seq_len is not None else self.seq_len
        self.dtype = dtype
        self.mode: Any = mode
        self._data: np.memmap[Any, Any] | None = None
        self._preload: np.ndarray[Any, Any] | None = None
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if not (1 <= self.min_seq_len <= self.seq_len):
            raise ValueError(
                f"min_seq_len must be in [1, seq_len={self.seq_len}], "
                f"got {self.min_seq_len}"
            )
        # Pre-load memmap into RAM for faster access
        if preload:
            self._preload = np.asarray(self.data)

    @property
    def block_size(self) -> int:
        return self.seq_len

    @property
    def data(self) -> np.memmap[Any, Any]:
        if self._data is None:
            self._data = np.memmap(
                self.path,
                dtype=self.dtype,
                mode=self.mode,
            )
        return self._data

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, index: int) -> np.ndarray[Any, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # Variable window: never exceed seq_len; never below min_seq_len.
        win = (
            int(np.random.randint(self.min_seq_len, self.seq_len + 1))
            if self.min_seq_len < self.seq_len
            else self.seq_len
        )
        end = index + win + 1
        if self._preload is not None:
            return np.asarray(self._preload[index:end], dtype=np.int64)
        return np.asarray(self.data[index:end], dtype=np.int64)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_data"] = None
        state["_preload"] = None
        return state
