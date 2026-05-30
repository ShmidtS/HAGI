import pytest


torch = pytest.importorskip("torch")

from hagi.data import PrefixLMBatch, create_prefix_lm_batch, prefix_lm_mask


def test_prefix_lm_mask_shape_and_dtype():
    mask = prefix_lm_mask([2, 2], total_len=6)

    assert tuple(mask.shape) == (6, 6)
    assert mask.dtype == torch.bool


def test_prefix_tokens_can_attend_to_all_prefix_tokens():
    mask = prefix_lm_mask([3], total_len=5)

    assert mask[:3, :3].all()
    assert not mask[:3, 3:].any()


def test_suffix_tokens_cannot_attend_to_other_samples():
    mask = prefix_lm_mask([2, 2], total_len=6)

    assert mask[2, 0]
    assert mask[2, 2]
    assert not mask[2, 3]
    assert not mask[5, 0]
    assert mask[5, 3]
    assert mask[5, 5]


def test_create_prefix_lm_batch_produces_mask_and_partition():
    batch = create_prefix_lm_batch([[1, 2, 3, 4], [5, 6, 7]], max_seq_len=5)

    assert isinstance(batch, PrefixLMBatch)
    assert tuple(batch.tokens.shape) == (2, 5)
    assert tuple(batch.mask.shape) == (2, 5, 5)
    assert batch.partition.tolist() == [2, 1]
    assert batch.tokens.tolist() == [[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]]
    assert batch.mask[0, :2, :2].all()
    assert not batch.mask[0, :2, 2:].any()
    assert batch.mask[0, 3, :4].all()
    assert not batch.mask[0, 3, 4]
