__version__ = "0.1.0"

# Filter benign CUDA-allocator warnings BEFORE any hagi submodule imports torch
# and triggers them (clifford._prime_caches moves tensors to CUDA at import,
# which fires "expandable_segments not supported" on Windows where the env-set
# expandable_segments:True allocator config is unsupported). Placing the filter
# here guarantees it is active before hagi.model.clifford is imported by any
# entry point (train.py, profile_steps.py, cli). The loop.py filter is kept as
# a second line of defense for scripts that import loop.py directly.
import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    message="expandable_segments not supported",
)
_warnings.filterwarnings(
    "ignore",
    message=r"Online softmax is disabled",
)
