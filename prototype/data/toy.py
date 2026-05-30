"""Tiny in-memory dataset for the overfit sanity test.

A fixed set of random short sequences. A correct training loop must drive loss
toward zero on this (the model has more than enough capacity to memorize it).
Used by tests/test_overfit.py — the cheapest possible check that the loop,
gradients, and optimizer are wired correctly.
"""

from __future__ import annotations

import torch


def make_toy_batch_fn(vocab_size: int, block_size: int, batch_size: int,
                      num_sequences: int = 8, device: str = "cpu", seed: int = 0):
    """Fixed toy corpus; get_batch() returns (x, y) drawn from it."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, vocab_size, (num_sequences, block_size + 1), generator=g)

    def get_batch():
        idx = torch.randint(0, num_sequences, (batch_size,), generator=g)
        chunk = data[idx]
        x = chunk[:, :-1].to(device)
        y = chunk[:, 1:].to(device)
        return x, y

    return get_batch
