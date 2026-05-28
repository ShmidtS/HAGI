"""HAGI model package."""

from .clifford import geometric_product, grade_projection, reverse
from .gdr import GradeConfig, GradeDecomposedRecurrence
from .hagi import HAGI, HAGIConfig
from .transformer import TransformerConfig

__all__ = [
    "HAGI",
    "HAGIConfig",
    "TransformerConfig",
    "GradeConfig",
    "GradeDecomposedRecurrence",
    "geometric_product",
    "grade_projection",
    "reverse",
]
