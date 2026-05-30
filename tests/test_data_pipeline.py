import numpy as np

from hagi.data import MemmapDataset, get_batch_memmap, get_batch_synthetic


def _shape(tensor):
    return tuple(tensor.shape)


def test_synthetic_batch_shapes():
    x, y = get_batch_synthetic(vocab_size=16, batch_size=4, seq_len=8)

    assert _shape(x) == (4, 8)
    assert _shape(y) == (4, 8)


def test_memmap_dataset_mock(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(32, dtype=np.uint16).tofile(path)
    dataset = MemmapDataset(path, block_size=8, dtype=np.uint16)

    assert len(dataset) == 24
    chunk = dataset[0]
    assert chunk.tolist() == list(range(9))

    x, y = get_batch_memmap(dataset, batch_size=2, seq_len=8)
    assert _shape(x) == (2, 8)
    assert _shape(y) == (2, 8)
