from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from hagi.data import MemmapDataset
from hagi.model import HAGI
from hagi.train.config import config_from_dict
from hagi.train.loop import LoopConfig, save_checkpoint, train
from hagi.train.optim import build_optimizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "rtx3070.yaml"
DEFAULT_CKPT_DIR = ROOT / "checkpoints" / "rtx3070"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def print_model_size(model: HAGI) -> None:
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp16_gb = params * 2 / 1024**3
    adamw_gb = params * 12 / 1024**3
    print(f"model parameters: total={params:,} trainable={trainable:,}")
    print(f"estimated VRAM: params_fp16={fp16_gb:.2f}GB adamw_training_state~={adamw_gb:.2f}GB")


def print_vram_usage() -> None:
    if not torch.cuda.is_available():
        print("VRAM unavailable: CUDA is not available")
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"VRAM after model creation: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")


def synthetic_batcher(vocab_size: int, batch_size: int, seq_len: int, device: str, generator: torch.Generator):
    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(vocab_size, (batch_size, seq_len), generator=generator, device=device)
        y = torch.randint(vocab_size, (batch_size, seq_len), generator=generator, device=device)
        return x, y

    return get_batch


def memmap_batcher(path: Path, batch_size: int, seq_len: int, device: str, dtype: str, generator: torch.Generator):
    dataset = MemmapDataset(path, block_size=seq_len, dtype=dtype)
    if len(dataset) <= 0:
        raise ValueError(f"memmap dataset is too small for seq_len={seq_len}: {path}")

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(len(dataset), (batch_size,), generator=generator).tolist()
        xs, ys = zip(*(dataset[index] for index in indices), strict=True)
        x = torch.tensor(np.array(xs), dtype=torch.long, device=device)
        y = torch.tensor(np.array(ys), dtype=torch.long, device=device)
        return x, y

    return get_batch


def build_batcher(cfg: dict[str, Any], device: str, train_path: Path | None, data_dir: Path, seq_len: int | None):
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    vocab_size = int(model_cfg.get("vocab_size", 49152))
    batch_size = int(train_cfg.get("batch_size", 1))
    seq_len = int(seq_len if seq_len is not None else data_cfg.get("max_seq_len", model_cfg.get("transformer", {}).get("max_seq_len", 2048)))
    seed = int(train_cfg.get("seed", 42))

    if train_path is None:
        configured_path = data_cfg.get("train_path") or data_cfg.get("path")
        train_path = Path(configured_path) if configured_path else None
    if train_path is None and data_dir.exists():
        bin_files = sorted(data_dir.glob("*.bin"))
        train_path = bin_files[0] if bin_files else None
    if train_path is not None and train_path.exists():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return memmap_batcher(train_path, batch_size, seq_len, device, data_cfg.get("dtype", "uint16"), generator)
    warnings.warn("no memmap .bin data found; falling back to synthetic data", RuntimeWarning, stacklevel=2)
    generator = torch.Generator(device=device if device.startswith("cuda") else "cpu")
    generator.manual_seed(seed)
    return synthetic_batcher(vocab_size, batch_size, seq_len, device, generator)


def build_loop_config(cfg: dict[str, Any], ckpt_dir: Path, max_steps: int | None) -> LoopConfig:
    train_cfg = cfg.get("training", {})
    return LoopConfig(
        max_steps=int(max_steps if max_steps is not None else train_cfg.get("max_steps", 100000)),
        warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
        learning_rate=float(train_cfg.get("learning_rate", 5.0e-4)),
        min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.1)),
        grad_accum_steps=int(train_cfg.get("grad_accum_steps", 16)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        precision=str(train_cfg.get("precision", "fp16")),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        eval_interval=int(train_cfg.get("eval_interval", 2000)),
        eval_iters=int(train_cfg.get("eval_iters", 50)),
        ckpt_interval=int(train_cfg.get("ckpt_interval", 5000)),
        ckpt_dir=str(ckpt_dir),
        log_interval=int(train_cfg.get("log_interval", 50)),
    )


def load_resume(model: HAGI, resume: Path, device: str) -> int:
    state = torch.load(resume, map_location=device, weights_only=True)
    if "model" in state:
        model.load_state_dict(state["model"])
        return int(state.get("step", 0))
    model.load_state_dict(state)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="train_rtx3070")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/fineweb_1M"))
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = config_from_dict(cfg.get("model", {}))
    model = HAGI(model_cfg).to(args.device)
    print_model_size(model)
    print_vram_usage()

    start_step = 0
    if args.resume is not None:
        start_step = load_resume(model, args.resume, args.device)
        print(f"resumed from step {start_step}")

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    optimizer = build_optimizer(model, cfg.get("training", {}))
    get_batch = build_batcher(cfg, args.device, args.train_path, args.data_dir, args.seq_len)
    loop_cfg = build_loop_config(cfg, args.ckpt_dir, args.max_steps)
    final_loss = train(model, optimizer, get_batch, loop_cfg, device=args.device)
    save_checkpoint(model, optimizer, loop_cfg.max_steps, str(args.ckpt_dir))
    print(f"final_loss {final_loss:.4f}")


if __name__ == "__main__":
    main()
