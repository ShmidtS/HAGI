import argparse
import json

import numpy as np
import pytest

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
    np.arange(32, dtype="uint16").tofile(path)
    dataset = MemmapDataset(path, block_size=8, dtype="uint16")

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
        array = np.memmap(paths[name], dtype="uint16", mode="r")
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


def test_v2_50m_preset_present_and_normalized():
    mix = download_data.parse_mix("v2_50m")
    assert abs(sum(mix.values()) - 1.0) < 1e-6
    assert set(mix) == {"edu", "cosmopedia", "wikitext", "smoltalk", "tinystories", "python_instruct", "openwebtext"}
    assert "tinycodes" not in mix  # nampdn-ai/tiny-codes is gated
    assert mix["edu"] == pytest.approx(0.5102, abs=1e-3)
    assert mix["wikitext"] == pytest.approx(0.1020, abs=1e-3)
    assert mix["openwebtext"] == pytest.approx(0.0306, abs=1e-3)


def test_new_presets_expose_open_hf_datasets():
    for key in ("tinystories", "wikitext", "openwebtext", "tinycodes"):
        assert key in download_data.DATASET_PRESETS, f"missing preset {key}"
        assert "dataset" in download_data.DATASET_PRESETS[key]


def test_row_text_handles_new_sources():
    assert download_data._row_text("tinystories", {"text": "Once upon a time"}) == "Once upon a time"
    assert download_data._row_text("wikitext", {"text": "Factual wiki text"}) == "Factual wiki text"
    assert download_data._row_text("openwebtext", {"text": "Web snippet"}) == "Web snippet"
    assert download_data._row_text("tinycodes", {"text": "print('hi')"}) == "print('hi')"
    assert download_data._row_text("tinycodes", {"code": "x = 1"}) == "x = 1"
    # tinycodes with neither text nor code yields empty string (filtered out).
    assert download_data._row_text("tinycodes", {"instruction": "noop", "output": "noop"}) == ""


def test_dataset_spec_for_new_sources():
    for source, expected in [
        ("tinystories", ("roneneldan/TinyStories", None, "train")),
        ("wikitext", ("Salesforce/wikitext", "wikitext-103-raw-v1", "train")),
        ("openwebtext", ("Skylion007/openwebtext", None, "train")),
        ("tinycodes", ("nampdn-ai/tiny-codes", None, "train")),
    ]:
        assert download_data._dataset_spec_for_source(source) == expected


def test_skip_existing_path_short_circuits(monkeypatch, tmp_path):
    args = argparse.Namespace(
        output=tmp_path,
        mix_ratios={"edu": 1.0},
        subset="1k",
        min_source_tokens=1024,
        min_length=50,
        skip_existing=True,
    )
    existing = tmp_path / "edu.bin"
    np.arange(4096, dtype="uint16").tofile(existing)

    calls = {"load_dataset": 0}

    def fake_load_dataset(*_args, **_kwargs):
        calls["load_dataset"] += 1
        raise AssertionError("load_dataset should not be called when --skip-existing and file present")

    monkeypatch.setattr(download_data, "load_dataset", fake_load_dataset, raising=False)
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    paths = download_data.download_mixed_token_bins(args)
    assert calls["load_dataset"] == 0
    assert paths["edu"] == existing


def test_parse_args_exposes_skip_existing():
    import sys
    saved = sys.argv
    try:
        sys.argv = ["download_data.py"]
        ns = download_data.parse_args()
        assert hasattr(ns, "skip_existing")
        assert ns.skip_existing is False
    finally:
        sys.argv = saved


def test_v3_presets_expose_open_hf_datasets():
    for key in ("wikipedia_en", "wikipedia_ru", "oscar_ru", "openwebmath"):
        assert key in download_data.DATASET_PRESETS, f"missing preset {key}"
        assert "dataset" in download_data.DATASET_PRESETS[key]
        assert download_data.DATASET_PRESETS[key]["dataset"]


def test_v3_presets_target_correct_sources():
    assert download_data.DATASET_PRESETS["wikipedia_en"]["dataset"] == "wikimedia/wikipedia"
    assert download_data.DATASET_PRESETS["wikipedia_en"]["name"] == "20231101.simple"
    assert download_data.DATASET_PRESETS["wikipedia_ru"]["dataset"] == "wikimedia/wikipedia"
    assert download_data.DATASET_PRESETS["wikipedia_ru"]["name"] == "20231101.ru"
    # OSCAR-2301 is gated; fall back to FineWeb-2 rus_Cyrl
    assert download_data.DATASET_PRESETS["oscar_ru"]["dataset"] == "HuggingFaceFW/fineweb-2"
    assert download_data.DATASET_PRESETS["oscar_ru"]["name"] == "rus_Cyrl"
    # openbmb/openwebmath is unavailable; use opc-fineweb-math-corpus
    assert "math" in str(download_data.DATASET_PRESETS["openwebmath"]["dataset"]).lower()


def test_v3_150m_preset_present_and_normalized():
    mix = download_data.parse_mix("v3_150m")
    assert abs(sum(mix.values()) - 1.0) < 1e-6
    expected_keys = {
        "edu",
        "cosmopedia",
        "wikipedia_en",
        "wikipedia_ru",
        "oscar_ru",
        "openwebmath",
        "smoltalk",
        "tinystories",
        "python_instruct",
        "openwebtext",
    }
    assert set(mix) == expected_keys
    # Russian coverage target ~17% (ru.wiki 10% + oscar_ru 6.67%)
    assert mix["wikipedia_ru"] + mix["oscar_ru"] == pytest.approx(0.1667, abs=1e-3)
    # Edu is still the largest single source
    assert mix["edu"] > mix["wikipedia_en"]


def test_dataset_spec_for_v3_sources():
    for source, expected_dataset in [
        ("wikipedia_en", "wikimedia/wikipedia"),
        ("wikipedia_ru", "wikimedia/wikipedia"),
        ("oscar_ru", "HuggingFaceFW/fineweb-2"),
    ]:
        spec = download_data._dataset_spec_for_source(source)
        assert spec[0] == expected_dataset


def test_v3_wikipedia_sources_use_different_configs():
    en_spec = download_data._dataset_spec_for_source("wikipedia_en")
    ru_spec = download_data._dataset_spec_for_source("wikipedia_ru")
    assert en_spec[1] == "20231101.simple"
    assert ru_spec[1] == "20231101.ru"
