from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable


@dataclass
class GoldenEvaluator:
    """Small evaluator for golden batches."""

    metric_name: str = "loss"

    def evaluate(
        self,
        model: Callable[..., Any],
        get_batch: Callable[[], Any],
        iters: int,
    ) -> dict[str, float]:
        if iters <= 0:
            return {self.metric_name: 0.0}

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for _ in range(iters):
            batch = get_batch()
            output = model(*batch) if isinstance(batch, tuple) else model(batch)
            metrics = self._metrics_from_output(output)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
                counts[name] = counts.get(name, 0) + 1

        return {name: totals[name] / counts[name] for name in totals}

    def _metrics_from_output(self, output: Any) -> dict[str, float]:
        if isinstance(output, dict):
            return {
                str(name): self._to_float(value)
                for name, value in output.items()
                if self._can_float(value)
            }
        return {self.metric_name: self._to_float(output)}

    @staticmethod
    def _can_float(value: Any) -> bool:
        try:
            GoldenEvaluator._to_float(value)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, Number):
            return float(value)
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)


def evaluate(
    model: Callable[..., Any],
    get_batch: Callable[[], Any],
    iters: int,
) -> dict[str, float]:
    return GoldenEvaluator().evaluate(model, get_batch, iters)
