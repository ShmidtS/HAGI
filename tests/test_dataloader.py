import numpy as np
import pytest

from hagi.data import MemmapDataset, get_memmap_dataloader


def test_memmap_dataset_creation_and_iteration(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(32, dtype="uint16").tofile(path)
    dataset = MemmapDataset(path, seq_len=8, dtype="uint16")

    assert len(dataset) == 24
    chunk = dataset[0]
    assert chunk.tolist() == list(range(9))


def test_memmap_dataloader_batch_shapes(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(64, dtype="uint16").tofile(path)
    loader = get_memmap_dataloader(path, batch_size=2, seq_len=8, num_workers=0, pin_memory=False)

    x, y = next(iter(loader))

    assert tuple(x.shape) == (2, 8)
    assert tuple(y.shape) == (2, 8)


def test_memmap_dataloader_shift_relationship(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(64, dtype="uint16").tofile(path)
    loader = get_memmap_dataloader(path, batch_size=4, seq_len=8, num_workers=0, pin_memory=False)

    x, y = next(iter(loader))

    assert (x[:, 1:] == y[:, :-1]).all()


def test_weighted_memmap_dataset_len_and_shift(tmp_path):
    from hagi.data import WeightedMemmapDataset

    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    np.arange(32, dtype="uint16").tofile(first)
    np.arange(100, 132, dtype="uint16").tofile(second)

    dataset = WeightedMemmapDataset(
        [(first, 0.75), (second, 0.25)],
        seq_len=8,
        dtype="uint16",
        seed=123,
    )

    assert len(dataset) == 48
    x, y = dataset[0]
    assert tuple(x.shape) == (8,)
    assert tuple(y.shape) == (8,)
    assert (x[1:] == y[:-1]).all()


def test_weighted_memmap_dataset_rejects_invalid_mix(tmp_path):
    from hagi.data import WeightedMemmapDataset

    path = tmp_path / "tokens.bin"
    np.arange(16, dtype="uint16").tofile(path)

    with pytest.raises(ValueError, match="mix must not be empty"):
        WeightedMemmapDataset([], seq_len=8)

    with pytest.raises(ValueError, match="weight must be positive and finite"):
        WeightedMemmapDataset([(path, 0.0)], seq_len=8)

    with pytest.raises(ValueError, match="weight must be positive and finite"):
        WeightedMemmapDataset([(path, np.nan)], seq_len=8)

    with pytest.raises(ValueError, match="weight must be positive and finite"):
        WeightedMemmapDataset([(path, np.inf)], seq_len=8)

    with pytest.raises(ValueError, match="seq_len must be positive"):
        WeightedMemmapDataset([(path, 1.0)], seq_len=0)


def test_weighted_memmap_dataset_rejects_too_small_source(tmp_path):
    from hagi.data import WeightedMemmapDataset

    path = tmp_path / "tokens.bin"
    np.arange(8, dtype="uint16").tofile(path)

    with pytest.raises(ValueError, match="source length must be greater than seq_len"):
        WeightedMemmapDataset([(path, 1.0)], seq_len=8)
