from __future__ import annotations

from dataclasses import dataclass

from hagi.utils import _clamp01


@dataclass(frozen=True, slots=True)
class BudgetValue:
    priority: float
    durability: float
    quality: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "priority", _clamp01(self.priority))
        object.__setattr__(self, "durability", _clamp01(self.durability))
        object.__setattr__(self, "quality", _clamp01(self.quality))

    def above_threshold(self, threshold: float) -> bool:
        threshold = _clamp01(threshold)
        return self.priority >= threshold and self.quality >= threshold


def merge_budgets(left: BudgetValue, right: BudgetValue) -> BudgetValue:
    return BudgetValue(
        max(left.priority, right.priority),
        max(left.durability, right.durability),
        max(left.quality, right.quality),
    )


def budget_decay(budget: BudgetValue, factor: float) -> BudgetValue:
    factor = _clamp01(factor)
    return BudgetValue(
        budget.priority * factor, budget.durability * factor, budget.quality
    )
