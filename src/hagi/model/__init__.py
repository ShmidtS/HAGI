"""Model components for HAGI."""

from .cast import CASTConfig, CASTHead, build_cast_targets
from .gdr import GradeConfig, GradeDecomposedRecurrence
from .hagi import HAGI, HAGIConfig
from .hdim_full import (
    DelayedHDIM,
    DomainRotor,
    DomainTransfer,
    GatedFusion,
    HDIMFull,
    HiddenToMultivector,
    InvariantExtractor,
)
from .hrm_full import HRMCore, HState, HTransition, LState, LTransition
from .msa import (
    DocumentWiseRoPE,
    HDIMSlotRouter,
    HostKvCache,
    MSAAttention,
    MemorySlot,
    SlotRegistry,
    SparseRouter,
)
from .moe import MoESwiGLU
from .transformer import RMSNorm, TransformerBlock, TransformerConfig, build_rope_cache

__all__ = [
    "HAGI",
    "HAGIConfig",
    "TransformerConfig",
    "TransformerBlock",
    "MoESwiGLU",
    "HRMCore",
    "HState",
    "LState",
    "HTransition",
    "LTransition",
    "GradeConfig",
    "GradeDecomposedRecurrence",
    "CASTConfig",
    "CASTHead",
    "build_cast_targets",
    "DelayedHDIM",
    "HDIMFull",
    "HiddenToMultivector",
    "DomainRotor",
    "InvariantExtractor",
    "DomainTransfer",
    "GatedFusion",
    "RMSNorm",
    "build_rope_cache",
    "MemorySlot",
    "SlotRegistry",
    "SparseRouter",
    "MSAAttention",
    "DocumentWiseRoPE",
    "HostKvCache",
    "HDIMSlotRouter",
]
