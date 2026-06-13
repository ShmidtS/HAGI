"""Sequential cycling dataset loader — train on one source for N cycles, then next."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    from torch.utils.data import DataLoader, Dataset as _TorchDataset, Sampler as _TorchSampler
    Dataset = cast(Any, _TorchDataset)
    Sampler = cast(Any, _TorchSampler)
except ImportError:
    DataLoader = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass

    class Sampler:  # type: ignore[no-redef]
        pass


def _shift_collate(batch: list[Any]) -> tuple[Any, Any]:
    array = np.stack([np.asarray(item, dtype=np.int64) for item in batch])
    x = array[:, :-1]
    y = array[:, 1:]
    from hagi.utils import _as_long_tensor
    return _as_long_tensor(x), _as_long_tensor(y)


class ChunkedRandomSampler(Sampler):
    """Random sampler that yields indices in small chunks to avoid OOM on randperm."""

    def __init__(self, data_source: Any, seed: int = 0, chunk_size: int = 4096):
        self.data_source = data_source
        self.seed = seed
        self.chunk_size = chunk_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        n = len(self.data_source)
        for i in range(0, n, self.chunk_size):
            chunk = min(self.chunk_size, n - i)
            indices = rng.integers(0, n, size=chunk)
            for idx in indices:
                yield int(idx)

    def __len__(self):
        return len(self.data_source)


class RandomSubsetDataset(Dataset):  # type: ignore[type-arg, misc]
    """Wraps a dataset and yields a random subset of a fixed size."""

    def __init__(self, base_dataset: Any, subset_size: int, seed: int = 0) -> None:
        self.base = base_dataset
        self.subset_size = subset_size
        self.seed = seed
        self._indices: np.ndarray | None = None

    def _build_indices(self) -> np.ndarray:
        if self._indices is None:
            rng = np.random.default_rng(self.seed)
            self._indices = rng.integers(0, len(self.base), size=self.subset_size)
        return self._indices

    def __len__(self) -> int:
        return self.subset_size

    def __getitem__(self, index: int) -> Any:
        indices = self._build_indices()
        return self.base[int(indices[index])]

    def __getstate__(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "subset_size": self.subset_size,
            "seed": self.seed,
            "_indices": self._indices,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.base = state["base"]
        self.subset_size = state["subset_size"]
        self.seed = state["seed"]
        self._indices = state.get("_indices")


class SequentialCyclingIterator:
    """Cycles through memmap datasets sequentially, N cycles per source.

    Each source is fully iterated (one epoch) per cycle when ``steps_per_cycle``
    is not set.  When ``steps_per_cycle`` is set, a random subset of that many
    batches is drawn per cycle.  After ``cycles_per_dataset`` cycles the iterator
    switches to the next source.  The iterator never raises ``StopIteration`` —
    it loops forever.
    """

    def __init__(
        self,
        entries: list[dict[str, Any]],
        batch_size: int,
        seq_len: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        dtype: str = "uint16",
        cycles_per_dataset: int = 1,
        steps_per_cycle: int | None = None,
    ):
        self.entries = entries
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dtype = dtype
        self.cycles_per_dataset = cycles_per_dataset
        self.steps_per_cycle = steps_per_cycle
        self.current_idx = 0
        self.current_cycle = 0
        self._current_iter: Any = None
        self._dataset_cache: dict[str, Any] = {}
        self._loader_cache: dict[tuple[int, int], Any] = {}

    def _make_loader(self, path: str | Path, seed: int = 0) -> Any:
        from hagi.data.dataloader import MemmapDataset
        path_key = str(path)
        if path_key not in self._dataset_cache:
            self._dataset_cache[path_key] = MemmapDataset(path, seq_len=self.seq_len, dtype=self.dtype, preload=True)
        base = self._dataset_cache[path_key]
        cache_key = (self.current_idx, self.current_cycle)
        if cache_key in self._loader_cache:
            return self._loader_cache[cache_key]
        if self.steps_per_cycle is not None and self.steps_per_cycle > 0:
            subset_size = self.steps_per_cycle * self.batch_size
            dataset = RandomSubsetDataset(base, subset_size, seed=seed)
            sampler = None
        else:
            dataset = base
            sampler = ChunkedRandomSampler(base, seed=seed)
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "collate_fn": _shift_collate,
            "drop_last": True,
        }
        if sampler is not None:
            kwargs["sampler"] = sampler
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 4
            kwargs["persistent_workers"] = True
        loader = DataLoader(dataset, **kwargs)  # type: ignore[operator]
        self._loader_cache[cache_key] = loader
        return loader

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Any, Any]:
        while True:
            if self._current_iter is None:
                self._advance()
            try:
                return next(self._current_iter)
            except StopIteration:
                self.current_cycle += 1
                if self.current_cycle >= self.cycles_per_dataset:
                    self.current_idx = (self.current_idx + 1) % len(self.entries)
                    self.current_cycle = 0
                self._current_iter = None

    def _advance(self) -> None:
        entry = self.entries[self.current_idx]
        path = entry["path"]
        name = entry.get("name", f"dataset_{self.current_idx}")
        seed = self.current_idx * 1000 + self.current_cycle
        self._current_iter = iter(self._make_loader(path, seed))
        print(f"[SequentialCycling] {name} (cycle {self.current_cycle + 1}/{self.cycles_per_dataset})")
