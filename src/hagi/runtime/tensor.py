from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class TensorWrapper:
    tensor: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        shape = getattr(self.tensor, "shape", ())
        return tuple(int(dim) for dim in shape)

    @property
    def dtype(self) -> Any:
        return getattr(self.tensor, "dtype", None)

    def to(self, *args: Any, **kwargs: Any) -> "TensorWrapper":
        return TensorWrapper(self.tensor.to(*args, **kwargs), self.metadata)
