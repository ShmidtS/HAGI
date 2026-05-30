from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PrefixLMBatch:
    tokens: torch.Tensor
    mask: torch.Tensor
    partition: torch.Tensor


def prefix_lm_mask(prefix_lengths: list[int], total_len: int) -> torch.Tensor:
    """Build a PrefixLM attention mask for contiguous packed samples."""
    if total_len < 0:
        raise ValueError("total_len must be non-negative")
    if any(length < 0 for length in prefix_lengths):
        raise ValueError("prefix lengths must be non-negative")
    if not prefix_lengths:
        return torch.zeros((total_len, total_len), dtype=torch.bool)
    if sum(prefix_lengths) > total_len:
        raise ValueError("prefix lengths exceed total length")

    mask = torch.zeros((total_len, total_len), dtype=torch.bool)
    remaining_suffix = total_len - sum(prefix_lengths)
    sample_count = len(prefix_lengths)
    suffix_lengths = [remaining_suffix // sample_count] * sample_count
    for index in range(remaining_suffix % sample_count):
        suffix_lengths[index] += 1

    start = 0
    for prefix_len, suffix_len in zip(prefix_lengths, suffix_lengths, strict=True):
        sample_len = prefix_len + suffix_len
        stop = start + sample_len
        if prefix_len > 0:
            mask[start : start + prefix_len, start : start + prefix_len] = True
        for query in range(start + prefix_len, stop):
            mask[query, start : query + 1] = True
        start = stop
    return mask


def _sample_prefix_lm_mask(sample_len: int, prefix_len: int, max_seq_len: int) -> torch.Tensor:
    mask = torch.zeros((max_seq_len, max_seq_len), dtype=torch.bool)
    if sample_len <= 0:
        return mask

    prefix_len = min(prefix_len, sample_len)
    if prefix_len > 0:
        mask[:prefix_len, :prefix_len] = True

    for query in range(prefix_len, sample_len):
        mask[query, : query + 1] = True
    return mask


def create_prefix_lm_batch(samples: list[list[int]], max_seq_len: int) -> PrefixLMBatch:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    batch_size = len(samples)
    tokens = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_seq_len, max_seq_len), dtype=torch.bool)
    partition = torch.zeros((batch_size,), dtype=torch.long)

    for index, sample in enumerate(samples):
        sample_len = min(len(sample), max_seq_len)
        if sample_len == 0:
            continue
        prefix_len = max(1, sample_len // 2)
        tokens[index, :sample_len] = torch.as_tensor(sample[:sample_len], dtype=torch.long)
        mask[index] = _sample_prefix_lm_mask(sample_len, prefix_len, max_seq_len)
        partition[index] = prefix_len

    return PrefixLMBatch(tokens=tokens, mask=mask, partition=partition)
