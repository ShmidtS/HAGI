"""Best-Fit Decreasing document packer for EOS-delimited sequences.

Per arch_decision §Data and audit_data P0-1: BFD packs whole documents into
fixed-length rows, separated by the EOS token, so the model never sees two
documents glued together without a separator. The packer pads short bins to
``seq_len`` with the EOS token to keep row alignment with the memmap
``MemmapDataset`` collate.

Algorithm: sort documents by length descending, then for each doc place it
into the bin with the smallest remaining capacity that still fits. New bins
are opened as needed. Output: list of length-``seq_len`` token lists, each
ending at the last EOS boundary (or padded with EOS).
"""

from __future__ import annotations

from typing import Iterable


def best_fit_decreasing_pack(
    docs: Iterable[list[int]],
    seq_len: int,
    eos_id: int,
    pad_short_bins: bool = True,
) -> list[list[int]]:
    """Pack documents into fixed-length sequences using best-fit decreasing.

    Args:
        docs: Iterable of documents, each a list[int] of token ids (no EOS).
        seq_len: Row length of the output packed bin.
        eos_id: End-of-sequence token id used to separate documents.
        pad_short_bins: If True, pad rows shorter than ``seq_len`` with ``eos_id``.

    Returns:
        List of length-``seq_len`` token lists.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    ordered = sorted(docs, key=len, reverse=True)
    bins: list[list[int]] = []
    bin_lengths: list[int] = []

    def _remaining(length: int) -> int:
        return seq_len - length

    for doc in ordered:
        if len(doc) >= seq_len:
            chunks = [doc[i : i + seq_len] for i in range(0, len(doc), seq_len)]
            for chunk in chunks:
                bins.append(list(chunk))
                bin_lengths.append(len(chunk))
            continue
        # Append EOS, then place.
        token_seq = list(doc) + [eos_id]
        best_idx = -1
        best_remaining = seq_len + 1
        for idx, length in enumerate(bin_lengths):
            remaining = _remaining(length)
            if remaining >= len(token_seq) and remaining < best_remaining:
                best_idx = idx
                best_remaining = remaining
        if best_idx >= 0:
            bins[best_idx].extend(token_seq)
            bin_lengths[best_idx] += len(token_seq)
        else:
            bins.append(token_seq)
            bin_lengths.append(len(token_seq))

    if pad_short_bins:
        for idx, length in enumerate(bin_lengths):
            if length < seq_len:
                bins[idx].extend([eos_id] * (seq_len - length))
                bin_lengths[idx] = seq_len

    return bins


__all__ = ["best_fit_decreasing_pack"]
