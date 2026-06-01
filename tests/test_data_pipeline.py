import argparse
import json

import numpy as np

from hagi.data import MemmapDataset, get_batch_memmap, get_batch_synthetic
import scripts.download_data as download_data
from scripts.download_data import materialize_token_bins, write_mix_manifest


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


def test_write_mix_manifest_includes_open_dataset_presets(tmp_path):
    write_mix_manifest(tmp_path, {"smollm_corpus": 0.8, "fineweb_edu_10bt": 0.2}, token_count=1000)

    data = json.loads((tmp_path / "mix.json").read_text(encoding="utf-8"))

    assert data["mix"] == {"smollm_corpus": 0.8, "fineweb_edu_10bt": 0.2}
    assert data["token_count"] == 1000
    assert data["presets"]["smollm_corpus"]["dataset"] == "HuggingFaceTB/smollm-corpus"
    assert data["presets"]["fineweb_edu_10bt"]["name"] == "sample-10BT"


def test_materialize_token_bins_writes_source_named_files(tmp_path):
    tokens_by_source = {
        "edu": [1, 2, 3, 4],
        "cosmopedia": [5, 6],
        "smoltalk": [7, 8],
        "python_instruct": [9, 10],
    }

    paths = materialize_token_bins(tmp_path, tokens_by_source)

    assert {path.name for path in paths.values()} == {"edu.bin", "cosmopedia.bin", "smoltalk.bin", "python_instruct.bin"}
    for name, tokens in tokens_by_source.items():
        array = np.memmap(paths[name], dtype=np.uint16, mode="r")
        assert array.tolist() == tokens

    manifest = json.loads((tmp_path / "download_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"]["edu"]["tokens"] == 4
    assert manifest["sources"]["python_instruct"]["path"] == "python_instruct.bin"


def test_main_materialize_mix_downloads_mixed_bins(monkeypatch, tmp_path):
    calls = []
    args = argparse.Namespace(
        output=tmp_path,
        mix="default",
        mix_ratios={},
        packing="bfd",
        subset="1k",
        materialize_mix=True,
        sft=False,
        dataset=None,
    )

    monkeypatch.setattr(download_data, "parse_args", lambda: args)
    monkeypatch.setattr(download_data, "write_mix_manifest", lambda *call_args, **call_kwargs: tmp_path / "mix.json")
    monkeypatch.setattr(download_data, "download_mixed_token_bins", lambda call_args: calls.append(dict(call_args.mix_ratios)))

    download_data.main()

    assert calls == [download_data.DEFAULT_MIX]
