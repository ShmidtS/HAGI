"""Data loading and processing.

- tokenizer.py : SmolLM2 (49K) tokenizer wrapper
- tokenize.py  : datatrove pipeline -> .bin shards (FineWeb-Edu + code + math)
- dataset.py   : memmap shard loader + get_batch (nanoGPT-style)
- toy.py       : in-memory toy corpus for the overfit sanity test

TODO (later stages):
  - PrefixLM packing (prefix bidirectional + response causal)
  - CoT distillation data ingestion (Stage 0 phase 2)
"""

from .dataset import MemmapTokenDataset, make_batch_fn
from .toy import make_toy_batch_fn

__all__ = ["MemmapTokenDataset", "make_batch_fn", "make_toy_batch_fn"]
