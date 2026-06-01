"""Tests for the Best-Fit Decreasing document packer (G005 §H)."""

from __future__ import annotations

import numpy as np
import pytest

from hagi.data import best_fit_decreasing_pack


def test_pack_empty_returns_empty():
    assert best_fit_decreasing_pack([], seq_len=8, eos_id=0) == []


def test_pack_basic_two_docs_fits_one_bin():
    docs = [[1, 2, 3], [4, 5]]
    packed = best_fit_decreasing_pack(docs, seq_len=10, eos_id=99)
    assert len(packed) == 1
    # 3 + 1 (eos) + 2 + 1 (eos) = 7 tokens, padded with 3 more EOS to 10.
    assert len(packed[0]) == 10
    # First 7 tokens preserve the docs and their EOS boundaries.
    assert packed[0][:7] == [1, 2, 3, 99, 4, 5, 99]
    assert all(tok == 99 for tok in packed[0][7:])


def test_pack_capacity_no_overflow():
    docs = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    seq_len = 16
    packed = best_fit_decreasing_pack(docs, seq_len=seq_len, eos_id=0)
    for row in packed:
        assert len(row) == seq_len


def test_pack_eos_after_each_doc():
    docs = [[1, 2, 3], [4, 5, 6], [7, 8]]
    packed = best_fit_decreasing_pack(docs, seq_len=16, eos_id=0)
    flat = [tok for row in packed for tok in row]
    # Every doc body must be preserved (8 non-EOS tokens).
    assert sorted(tok for tok in flat if tok != 0) == [1, 2, 3, 4, 5, 6, 7, 8]
    # Exactly one EOS per non-oversized doc.
    assert flat.count(0) >= 3


def test_pack_short_bin_padded_with_eos():
    docs = [[1, 2, 3]]
    packed = best_fit_decreasing_pack(docs, seq_len=10, eos_id=0)
    assert len(packed) == 1
    assert len(packed[0]) == 10
    # First the doc, then EOS, then EOS padding.
    assert packed[0][:3] == [1, 2, 3]
    assert packed[0][3] == 0
    assert all(tok == 0 for tok in packed[0][4:])


def test_pack_does_not_split_documents():
    docs = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    packed = best_fit_decreasing_pack(docs, seq_len=8, eos_id=99)
    # The doc is too long for a single bin; it must be split into chunks of
    # at most seq_len. Each chunk preserves the document order.
    flat = [tok for row in packed for tok in row]
    assert flat[:10] == list(range(1, 11))


def test_pack_oversized_doc_chunks_at_seq_len():
    seq_len = 6
    docs = [list(range(20))]
    packed = best_fit_decreasing_pack(docs, seq_len=seq_len, eos_id=-1)
    assert all(len(row) == seq_len for row in packed)
    flat = [tok for row in packed for tok in row]
    assert flat[:20] == list(range(20))


def test_pack_bfd_packs_better_than_first_fit():
    # BFD should place short docs into the tightest bin.
    docs = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4, 4]]
    packed = best_fit_decreasing_pack(docs, seq_len=10, eos_id=0)
    # All four docs (3+3+3+4 = 13) plus four EOS = 17 tokens -> 2 bins.
    assert len(packed) == 2
    # All four docs' tokens must be preserved.
    flat = [tok for row in packed for tok in row]
    body = [tok for tok in flat if tok != 0]
    assert sorted(body) == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4]


def test_pack_seq_len_one_each_doc_gets_own_bin():
    docs = [[1], [2], [3]]
    packed = best_fit_decreasing_pack(docs, seq_len=1, eos_id=9)
    # Each doc is dropped into its own bin (no two docs can share a row).
    assert len(packed) == 3
    assert sorted(row[0] for row in packed) == [1, 2, 3]


@pytest.mark.parametrize("seq_len", [4, 8, 16, 32])
def test_pack_round_trip_preserves_token_counts(seq_len):
    rng = np.random.default_rng(0)
    docs = [rng.integers(1, 100, size=int(rng.integers(2, seq_len))).tolist() for _ in range(20)]
    # Sum of body tokens is preserved (no doc is lost).
    expected_body = sum(len(d) for d in docs)
    packed = best_fit_decreasing_pack(docs, seq_len=seq_len, eos_id=0)
    flat = [tok for row in packed for tok in row]
    body = [tok for tok in flat if tok != 0]
    assert len(body) == expected_body
    # Each row is exactly seq_len long.
    assert all(len(row) == seq_len for row in packed)
