from __future__ import annotations

from typing import Any, Iterable

from hagi.nars.task import Task

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None  # type: ignore[assignment]

import numpy as np


class MSAAdapter:
    """Sparse attention routing from NARS task priorities."""

    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k

    def build_mask(self, tasks: Iterable[Task], seq_len: int) -> Any:
        task_list = list(tasks)
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if torch is not None:
            mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
        else:
            mask = np.zeros((seq_len, seq_len), dtype=bool)
        if seq_len == 0:
            return mask

        selected = self._selected_indices(task_list, seq_len)
        if not selected:
            return mask
        if torch is not None:
            index = torch.tensor(selected, dtype=torch.long)
            mask[:, index] = True
        else:
            mask[:, selected] = True
        return mask

    def route_kv(self, tasks: Iterable[Task], k: Any, v: Any) -> tuple[Any, Any]:
        seq_len = int(k.shape[-2])
        selected = self._selected_indices(list(tasks), seq_len)
        if not selected:
            return k[..., :0, :], v[..., :0, :]
        if torch is not None and isinstance(k, torch.Tensor):
            index = torch.tensor(selected, dtype=torch.long, device=k.device)
            return k.index_select(-2, index), v.index_select(-2, index.to(v.device))
        return np.take(np.asarray(k), selected, axis=-2), np.take(np.asarray(v), selected, axis=-2)

    def _selected_indices(self, tasks: list[Task], seq_len: int) -> list[int]:
        if seq_len <= 0 or not tasks:
            return []
        limit = min(seq_len, self.top_k if self.top_k is not None else len(tasks))
        ranked = sorted(
            enumerate(tasks),
            key=lambda item: (-item[1].budget.priority, item[0]),
        )
        return sorted(index % seq_len for index, _task in ranked[:limit])
