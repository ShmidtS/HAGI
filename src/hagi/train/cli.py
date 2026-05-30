"""Command-line entry point for HAGI training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

try:
    import typer
except ImportError:  # pragma: no cover - dependency fallback
    typer = None

from hagi.data import MemmapDataset
from hagi.model import HAGI
from hagi.train.config import config_from_dict
from hagi.train.loop import LoopConfig, train
from hagi.train.optim import build_optimizer


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _tiny_batcher(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: str,
    generator: torch.Generator,
):
    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(vocab_size, (batch_size, seq_len), generator=generator).to(device)
        y = torch.randint(vocab_size, (batch_size, seq_len), generator=generator).to(device)
        return x, y

    return get_batch


def _memmap_batcher(
    dataset: MemmapDataset,
    batch_size: int,
    device: str,
    generator: torch.Generator,
):
    if len(dataset) <= 0:
        raise ValueError(f"memmap dataset is too small for block_size={dataset.block_size}")

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(len(dataset), (batch_size,), generator=generator).tolist()
        xs, ys = zip(*(dataset[i] for i in idx), strict=True)
        x = torch.tensor(xs, dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        return x, y

    return get_batch


def _build_batcher(
    cfg: dict[str, Any],
    device: str,
    overfit: bool,
    generator: torch.Generator,
):
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    vocab_size = int(model_cfg.get("vocab_size", 32000))
    batch_size = int(train_cfg.get("batch_size", 4))
    seq_len = int(data_cfg.get("max_seq_len", model_cfg.get("transformer", {}).get("max_seq_len", 128)))

    if overfit:
        return _tiny_batcher(vocab_size, min(batch_size, 4), min(seq_len, 64), device, generator)

    train_path = data_cfg.get("train_path") or data_cfg.get("path")
    if train_path:
        block_size = int(data_cfg.get("block_size", seq_len))
        dtype = data_cfg.get("dtype", "uint16")
        dataset = MemmapDataset(train_path, block_size=block_size, dtype=dtype)
        return _memmap_batcher(dataset, batch_size, device, generator)

    return _tiny_batcher(vocab_size, min(batch_size, 4), min(seq_len, 64), device, generator)


def run(
    config: Path,
    device: str = "cpu",
    precision: str = "bf16",
    resume: Path | None = None,
    overfit: bool = False,
    max_steps: int | None = None,
    seed: int = 42,
    ckpt_dir: str = "checkpoints",
) -> None:
    cfg = _load_yaml(config)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    model_cfg = config_from_dict(cfg.get("model", {}))
    model = HAGI(model_cfg)
    start_step = 0
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=True)
        if "model" in state:
            model.load_state_dict(state["model"])
            start_step = int(state.get("step", 0))
        else:
            model.load_state_dict(state)

    train_cfg = dict(cfg.get("training", {}))
    train_cfg["precision"] = precision
    train_cfg["ckpt_dir"] = ckpt_dir
    if max_steps is not None:
        train_cfg["max_steps"] = max_steps
    loop_cfg = LoopConfig(
        max_steps=int(train_cfg.get("max_steps", 50000)),
        warmup_steps=int(train_cfg.get("warmup_steps", 2000)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.1)),
        grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        precision=str(train_cfg.get("precision", precision)),
        eval_interval=int(cfg.get("eval", {}).get("every_n_steps", train_cfg.get("eval_interval", 2000))),
        eval_iters=int(train_cfg.get("eval_iters", 50)),
        ckpt_interval=int(train_cfg.get("ckpt_interval", 5000)),
        ckpt_dir=str(train_cfg.get("ckpt_dir", ckpt_dir)),
        log_interval=int(train_cfg.get("log_interval", 50)),
    )
    optimizer = build_optimizer(model, train_cfg)
    get_batch = _build_batcher(cfg, device, overfit, generator)

    if start_step:
        print(f"resumed from step {start_step}")
    loss = train(model, optimizer, get_batch, loop_cfg, device=device)
    print(f"final_loss {loss:.4f}")


def main() -> None:
    if typer is not None:
        def cli(
            config: Path = typer.Option(..., "--config"),
            device: str = typer.Option("cpu", "--device"),
            precision: str = typer.Option("bf16", "--precision"),
            resume: Path | None = typer.Option(None, "--resume"),
            overfit: bool = typer.Option(False, "--overfit"),
            max_steps: int | None = typer.Option(None, "--max-steps"),
            seed: int = typer.Option(42, "--seed"),
            ckpt_dir: str = typer.Option("checkpoints", "--ckpt-dir"),
        ) -> None:
            run(config, device, precision, resume, overfit, max_steps, seed, ckpt_dir)

        typer.run(cli)
        return

    import argparse

    parser = argparse.ArgumentParser(prog="hagi-train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
