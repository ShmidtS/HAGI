import numpy as np

from hagi.data import MemmapDataset, get_memmap_dataloader


def test_memmap_dataset_creation_and_iteration(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(32, dtype=np.uint16).tofile(path)
    dataset = MemmapDataset(path, seq_len=8, dtype=np.uint16)

    assert len(dataset) == 24
    chunk = dataset[0]
    assert chunk.tolist() == list(range(9))


def test_memmap_dataloader_batch_shapes(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(64, dtype=np.uint16).tofile(path)
    loader = get_memmap_dataloader(path, batch_size=2, seq_len=8, num_workers=0, pin_memory=False)

    x, y = next(iter(loader))

    assert tuple(x.shape) == (2, 8)
    assert tuple(y.shape) == (2, 8)


def test_memmap_dataloader_shift_relationship(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(64, dtype=np.uint16).tofile(path)
    loader = get_memmap_dataloader(path, batch_size=4, seq_len=8, num_workers=0, pin_memory=False)

    x, y = next(iter(loader))

    assert (x[:, 1:] == y[:, :-1]).all()
