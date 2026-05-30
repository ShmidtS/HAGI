from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class MemmapDataset:
    def __init__(
        self,
        path: str | Path,
        block_size: int,
        dtype: str | np.dtype[Any] = np.uint16,
        mode: str = "r",
    ) -> None:
        self.path = Path(path)
        self.block_size = block_size
        self.data = np.memmap(self.path, dtype=dtype, mode=mode)

    def __len__(self) -> int:
        if self.block_size <= 0:
            return 0
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, index: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        chunk = np.asarray(self.data[index : index + self.block_size + 1], dtype=np.int64)
        return chunk[:-1], chunk[1:]
