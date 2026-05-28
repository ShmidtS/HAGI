"""Memmap binary dataset + batch sampler (nanoGPT-style).

Tokenized corpora are stored as flat uint16/uint32 .bin shards (token id stream).
`get_batch` samples random contiguous windows of length `block_size` and returns
(x, y) where y is x shifted by one. This is the fast path for pretraining: no
per-item Python overhead, just memmap slicing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class MemmapTokenDataset:
    """Wraps a directory of .bin shards as one logical token stream."""

    def __init__(self, data_dir: str | Path, dtype: str = "uint16"):
        self.data_dir = Path(data_dir)
        self.dtype = np.dtype(dtype)
        self.shards = sorted(self.data_dir.glob("*.bin"))
        if not self.shards:
            raise FileNotFoundError(f"no .bin shards in {self.data_dir}")
        # One memmap per shard; concatenation is logical (sample within a shard).
        self._mmaps = [np.memmap(s, dtype=self.dtype, mode="r") for s in self.shards]
        self._lengths = [len(m) for m in self._mmaps]

    def get_batch(self, batch_size: int, block_size: int, device: str = "cpu",
                  generator: np.random.Generator | None = None):
        rng = generator or np.random
        xs, ys = [], []
        for _ in range(batch_size):
            si = int(rng.integers(len(self._mmaps)))
            m = self._mmaps[si]
            hi = self._lengths[si] - block_size - 1
            if hi <= 0:
                raise ValueError(f"shard {self.shards[si]} shorter than block_size+1")
            start = int(rng.integers(hi))
            chunk = m[start : start + block_size + 1].astype(np.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def make_batch_fn(data_dir: str | Path, batch_size: int, block_size: int,
                  device: str = "cpu", seed: int = 0, dtype: str = "uint16"):
    """Return a zero-arg get_batch() closure for the training loop."""
    ds = MemmapTokenDataset(data_dir, dtype=dtype)
    gen = np.random.default_rng(seed)

    def get_batch():
        return ds.get_batch(batch_size, block_size, device=device, generator=gen)

    return get_batch
