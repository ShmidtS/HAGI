from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

UINT16_MAX = 65535


def write_shards(
    token_batches: Iterable[np.ndarray],
    output_dir: str | Path,
    shard_size: int = 100_000_000,
    dtype=np.uint16,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buf: list[np.ndarray] = []
    buf_len = 0
    idx = 0

    def flush():
        nonlocal buf, buf_len, idx
        if buf_len == 0:
            return
        arr = np.concatenate(buf).astype(dtype, copy=False)
        path = out / f"shard_{idx:05d}.bin"
        arr.tofile(path)
        paths.append(path)
        idx += 1
        buf, buf_len = [], 0

    for tb in token_batches:
        a = np.asarray(tb)
        if a.size == 0:
            continue
        buf.append(a)
        buf_len += a.size
        if buf_len >= shard_size:
            flush()
    flush()
    return paths
