"""Model components for HAGI."""

from .binary import BinaryFactorizedLinear, BinarySwiGLU
from .clifford_router import CliffordExpertRouter
from .gdr import GradeConfig, GradeDecomposedRecurrence
from .hagi import HAGI, HAGIConfig
from .hdim_full import DelayedHDIM, DomainRotor, DomainTransfer, GatedFusion, HDIMFull, HiddenToMultivector, InvariantExtractor
from .hrm_full import HRMCore, HState, HTransition, LState, LTransition
from .moe import MoEBinarySwiGLU, MoEOutput
from .transformer import RMSNorm, TransformerBlock, TransformerConfig, build_rope_cache

__all__ = [
    "BinaryFactorizedLinear",
    "BinarySwiGLU",
    "CliffordExpertRouter",
    "HAGI",
    "HAGIConfig",
    "TransformerConfig",
    "TransformerBlock",
    "HRMCore",
    "HState",
    "LState",
    "HTransition",
    "LTransition",
    "GradeConfig",
    "GradeDecomposedRecurrence",
    "DelayedHDIM",
    "HDIMFull",
    "HiddenToMultivector",
    "DomainRotor",
    "MoEBinarySwiGLU",
    "MoEOutput",
    "InvariantExtractor",
    "DomainTransfer",
    "GatedFusion",
    "RMSNorm",
    "build_rope_cache",
]
