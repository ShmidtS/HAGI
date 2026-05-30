from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


T = TypeVar("T")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _item_key(item: Any) -> str:
    if hasattr(item, "term"):
        return repr(item.term)
    if hasattr(item, "sentence") and hasattr(item.sentence, "term"):
        return repr(item.sentence.term)
    return repr(item)


@dataclass(slots=True)
class Bag(Generic[T]):
    items: dict[str, T] = field(default_factory=dict)
    _priorities: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _sequence: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _counter: int = field(default=0, init=False, repr=False)

    def put(self, item: T, priority: float) -> None:
        key = _item_key(item)
        self.items[key] = item
        self._priorities[key] = _clamp01(priority)
        self._sequence[key] = self._counter
        self._counter += 1

    def take(self) -> T | None:
        if not self.items:
            return None
        key = min(self.items, key=lambda name: (-self._priorities[name], self._sequence[name], name))
        item = self.items.pop(key)
        self._priorities.pop(key, None)
        self._sequence.pop(key, None)
        return item

    def get(self, name: str) -> T | None:
        return self.items.get(name)

    def priority(self, name: str) -> float | None:
        return self._priorities.get(name)

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)
