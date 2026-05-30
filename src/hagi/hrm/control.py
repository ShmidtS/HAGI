from __future__ import annotations

from typing import Any, Iterable

from hagi.nars.concept import Concept

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None  # type: ignore[assignment]

import numpy as np


class HRMController:
    """Route hidden states using NARS concept budget priorities."""

    def route(self, concepts: Iterable[Concept], hidden_states: Any) -> Any:
        concept_list = list(concepts)
        if not concept_list:
            return hidden_states

        priorities = [concept.budget.priority for concept in concept_list]
        if torch is not None and isinstance(hidden_states, torch.Tensor):
            weights = self._torch_softmax(priorities, hidden_states)
            return self._apply_weights(hidden_states, weights)

        array = np.asarray(hidden_states)
        weights = self._numpy_softmax(priorities, array.dtype)
        return self._apply_weights(array, weights)

    def _apply_weights(self, hidden_states: Any, weights: Any) -> Any:
        state_count = int(hidden_states.shape[0])
        if state_count == 0:
            return hidden_states

        if len(weights) < state_count:
            repeats = (state_count + len(weights) - 1) // len(weights)
            if torch is not None and isinstance(hidden_states, torch.Tensor):
                weights = weights.repeat(repeats)
            else:
                weights = np.tile(weights, repeats)
        weights = weights[:state_count]
        while weights.ndim < hidden_states.ndim:
            weights = weights.unsqueeze(-1) if torch is not None and isinstance(hidden_states, torch.Tensor) else np.expand_dims(weights, -1)
        return hidden_states * weights

    def _torch_softmax(self, priorities: list[float], hidden_states: Any) -> Any:
        values = torch.tensor(priorities, dtype=hidden_states.dtype, device=hidden_states.device)
        return torch.softmax(values, dim=0)

    def _numpy_softmax(self, priorities: list[float], dtype: Any) -> np.ndarray:
        values = np.asarray(priorities, dtype=np.float64)
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        weights = exp / np.sum(exp)
        return weights.astype(dtype if np.issubdtype(dtype, np.floating) else np.float64)
