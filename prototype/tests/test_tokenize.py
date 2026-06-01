"""Tokenizer shard-format tests.

write_shards must produce exactly what MemmapTokenDataset reads — the seam the
data-pipeline dry-run exposed (datatrove wrote `.ds`; the loader globs `.bin`).
"""

import numpy as np

from prototype.data.dataset import MemmapTokenDataset
from prototype.data.tokenize import write_shards


def test_write_shards_roundtrip_through_loader(tmp_path):
    batches = [
        np.array([1, 2, 3, 0], dtype=np.int64),
        np.array([4, 5, 0], dtype=np.int64),
        np.array([6, 7, 8, 9, 0], dtype=np.int64),
    ]
    paths = write_shards(batches, tmp_path, shard_size=4)  # small -> multiple shards
    assert len(paths) >= 2

    ds = MemmapTokenDataset(tmp_path, dtype="uint16")
    allvals = np.concatenate([np.asarray(m) for m in ds._mmaps])
    assert allvals.tolist() == [1, 2, 3, 0, 4, 5, 0, 6, 7, 8, 9, 0]


def test_write_shards_uint16_holds_smollm_vocab(tmp_path):
    # SmolLM2 max id (49151) must round-trip through uint16 shards intact.
    paths = write_shards([np.array([49151, 0, 100], dtype=np.int64)], tmp_path)
    arr = np.fromfile(paths[0], dtype=np.uint16)
    assert arr.tolist() == [49151, 0, 100]
