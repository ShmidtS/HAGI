import argparse
import ast
import contextlib
import sys

import numpy as np
import pytest
import torch

import scripts.train as train_script
from scripts.train import build_full_dataloader, main, resolve_mix_paths, resolve_train_path, run_dry_profile


def test_resolve_train_path_calls_use_supported_positional_args():
    tree = ast.parse((__import__("pathlib").Path(__file__).parents[1] / "scripts" / "train.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "resolve_train_path"
    ]

    assert calls
    assert all(len(call.args) in {2, 3} for call in calls)


def test_resolve_train_path_supports_legacy_and_override_arities(tmp_path):
    fallback = tmp_path / "fallback.bin"
    override = tmp_path / "override.bin"
    fallback.write_bytes(b"\x00\x00")

    assert resolve_train_path({}, tmp_path) == fallback
    assert resolve_train_path({}, override, tmp_path) == override


def test_resolve_train_path_override_wins_over_config(tmp_path):
    override = tmp_path / "override.bin"
    cfg = {"data": {"train_path": "configured.bin"}}

    assert resolve_train_path(cfg, override, tmp_path) == override


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.proj = torch.nn.Linear(8, 32)

    def forward(self, x):
        return self.proj(self.embedding(x))


def test_run_dry_profile_reports_without_optimizer_step(capsys):
    model = TinyModel()
    batch = (torch.randint(0, 32, (2, 4)), torch.randint(0, 32, (2, 4)))

    run_dry_profile(model, [batch], "cpu", use_prefix_lm=False, precision="fp32")

    output = capsys.readouterr().out
    assert "dry_run_loss" in output
    assert "dry_run_complete no optimizer step executed" in output


def test_run_dry_profile_uses_configured_precision(monkeypatch):
    model = TinyModel()
    batch = (torch.randint(0, 32, (2, 4)), torch.randint(0, 32, (2, 4)))
    calls = []

    @contextlib.contextmanager
    def fake_autocast_ctx(precision, device):
        calls.append((precision, device))
        yield

    monkeypatch.setattr(train_script, "autocast_ctx", fake_autocast_ctx)

    run_dry_profile(model, [batch], "cpu", use_prefix_lm=False, precision="bf16")

    assert calls == [("bf16", "cpu")]


def test_run_full_dry_run_exits_before_build_optimizer(tmp_path, monkeypatch):
    class FakeHAGI(TinyModel):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

        def num_parameters(self):
            return sum(param.numel() for param in self.parameters())

    batch = (torch.randint(0, 32, (2, 4)), torch.randint(0, 32, (2, 4)))
    args = argparse.Namespace(
        learning_rate=None,
        device="cpu",
        resume=None,
        train_path=None,
        data_dir=tmp_path,
        dataset_mode=None,
        max_steps=None,
        dry_run=True,
        ckpt_dir=tmp_path / "ckpt",
    )
    cfg = {
        "model": {
            "vocab_size": 32,
            "hidden_size": 8,
            "perception_layers": 1,
            "reasoning_layers": 1,
            "expression_layers": 1,
            "use_gdr": False,
            "transformer": {"hidden_size": 8, "num_query_heads": 1, "num_kv_heads": 1, "intermediate_size": 16},
        },
        "training": {"batch_size": 2, "precision": "fp32"},
        "data": {"max_seq_len": 4, "mix_paths": [{"path": "dummy.bin", "weight": 1.0}]},
    }

    def fail_build_optimizer(*args, **kwargs):
        raise AssertionError("build_optimizer should not be called during dry-run")

    monkeypatch.setattr(train_script, "HAGI", FakeHAGI)
    monkeypatch.setattr(train_script, "build_optimizer", fail_build_optimizer)
    monkeypatch.setattr(train_script, "build_full_dataloader", lambda *args, **kwargs: ([batch], None, 2, 4, False))

    train_script.run_full(args, cfg)


def test_run_full_passes_explicit_train_path_over_mix_paths(tmp_path, monkeypatch):
    class FakeHAGI(TinyModel):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

        def num_parameters(self):
            return sum(param.numel() for param in self.parameters())

    batch = (torch.randint(0, 32, (2, 4)), torch.randint(0, 32, (2, 4)))
    override = tmp_path / "override.bin"
    args = argparse.Namespace(
        learning_rate=None,
        device="cpu",
        resume=None,
        train_path=override,
        data_dir=tmp_path,
        dataset_mode=None,
        max_steps=None,
        dry_run=True,
        ckpt_dir=tmp_path / "ckpt",
    )
    cfg = {
        "model": {
            "vocab_size": 32,
            "hidden_size": 8,
            "perception_layers": 1,
            "reasoning_layers": 1,
            "expression_layers": 1,
            "use_gdr": False,
            "transformer": {"hidden_size": 8, "num_query_heads": 1, "num_kv_heads": 1, "intermediate_size": 16},
        },
        "training": {"batch_size": 2, "precision": "fp32"},
        "data": {"max_seq_len": 4, "mix_paths": [{"path": "mixed.bin", "weight": 1.0}]},
    }
    captured = {}

    def fake_build_full_dataloader(cfg, train_path, *args, **kwargs):
        captured["train_path"] = train_path
        return [batch], None, 2, 4, False

    monkeypatch.setattr(train_script, "HAGI", FakeHAGI)
    monkeypatch.setattr(train_script, "build_full_dataloader", fake_build_full_dataloader)

    train_script.run_full(args, cfg)

    assert captured["train_path"] == override


def test_build_full_dataloader_uses_explicit_train_path_over_mix_paths(tmp_path):
    override = tmp_path / "override.bin"
    np.arange(32, dtype=np.uint16).tofile(override)
    cfg = {
        "training": {"batch_size": 2, "seed": 123},
        "data": {
            "max_seq_len": 8,
            "num_workers": 0,
            "pin_memory": False,
            "mix_paths": [{"path": "missing.bin", "weight": 1.0}],
        },
    }

    train_loader, eval_loader, batch_size, seq_len, pin_memory = build_full_dataloader(
        cfg,
        train_path=override,
        data_dir=tmp_path,
        use_prefix_lm=True,
        device="cpu",
        eval_samples=0,
        dataset_mode="memmap",
    )

    assert train_loader is not None
    assert eval_loader is None
    assert batch_size == 2
    assert seq_len == 8
    assert pin_memory is False


def test_resolve_mix_paths_resolves_relative_paths(tmp_path):
    data_cfg = {"mix_paths": [{"path": "edu.bin", "weight": 0.7}]}

    assert resolve_mix_paths(data_cfg, tmp_path) == [(tmp_path / "edu.bin", 0.7)]


def test_build_full_dataloader_accepts_memmap_packed_mix_paths(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    np.arange(32, dtype=np.uint16).tofile(first)
    np.arange(100, 132, dtype=np.uint16).tofile(second)
    cfg = {
        "training": {"batch_size": 2, "seed": 123},
        "data": {
            "max_seq_len": 8,
            "num_workers": 0,
            "pin_memory": False,
            "mix_paths": [
                {"path": "first.bin", "weight": 0.75},
                {"path": "second.bin", "weight": 0.25},
            ],
        },
    }

    train_loader, eval_loader, batch_size, seq_len, pin_memory = build_full_dataloader(
        cfg,
        train_path=None,
        data_dir=tmp_path,
        use_prefix_lm=False,
        device="cpu",
        eval_samples=4,
        dataset_mode="memmap_packed",
    )
    x, y = next(iter(train_loader))

    assert eval_loader is None
    assert batch_size == 2
    assert seq_len == 8
    assert pin_memory is False
    assert tuple(x.shape) == (2, 8)
    assert (x[:, 1:] == y[:, :-1]).all()


def test_build_full_dataloader_uses_eval_path_with_mix_paths(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    eval_path = tmp_path / "eval.bin"
    np.arange(32, dtype=np.uint16).tofile(first)
    np.arange(100, 132, dtype=np.uint16).tofile(second)
    np.arange(200, 232, dtype=np.uint16).tofile(eval_path)
    cfg = {
        "training": {"batch_size": 2, "seed": 123},
        "data": {
            "max_seq_len": 8,
            "num_workers": 0,
            "pin_memory": False,
            "mix_paths": [
                {"path": "first.bin", "weight": 0.75},
                {"path": "second.bin", "weight": 0.25},
            ],
            "eval_path": "eval.bin",
        },
    }

    train_loader, eval_loader, batch_size, seq_len, pin_memory = build_full_dataloader(
        cfg,
        train_path=None,
        data_dir=tmp_path,
        use_prefix_lm=False,
        device="cpu",
        eval_samples=0,
        dataset_mode="memmap_packed",
    )
    assert train_loader is not None
    assert eval_loader is not None
    train_x, train_y = next(iter(train_loader))
    eval_x, eval_y = next(iter(eval_loader))
    assert batch_size == 2
    assert seq_len == 8
    assert pin_memory is False
    assert tuple(train_x.shape) == (2, 8)
    assert tuple(eval_x.shape) == (2, 8)
    assert (train_x[:, 1:] == train_y[:, :-1]).all()
    assert (eval_x[:, 1:] == eval_y[:, :-1]).all()


@pytest.mark.parametrize("mode", ["auto", "basic", "fast"])
def test_main_rejects_dry_run_outside_full_mode(tmp_path, monkeypatch, mode):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  vocab_size: 32\n"
        "training:\n"
        "  batch_size: 1\n"
        "data:\n"
        "  max_seq_len: 8\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--config", str(config_path), "--device", "cpu", "--dry-run", "--mode", mode],
    )

    with pytest.raises(ValueError, match="--dry-run is only supported in full mode"):
        main()


def test_build_full_dataloader_rejects_mix_paths_with_prefix_lm(tmp_path):
    first = tmp_path / "first.bin"
    np.arange(32, dtype=np.uint16).tofile(first)
    cfg = {
        "training": {"batch_size": 2, "seed": 123},
        "data": {
            "max_seq_len": 8,
            "num_workers": 0,
            "pin_memory": False,
            "mix_paths": [{"path": "first.bin", "weight": 1.0}],
        },
    }

    with pytest.raises(ValueError, match="mix_paths does not support prefix_lm"):
        build_full_dataloader(
            cfg,
            train_path=None,
            data_dir=tmp_path,
            use_prefix_lm=True,
            device="cpu",
            eval_samples=0,
            dataset_mode="memmap",
        )
