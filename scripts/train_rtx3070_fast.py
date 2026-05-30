from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from hagi.data import get_memmap_dataloader
from hagi.model import HAGI
from hagi.train.config import config_from_dict
from hagi.train.loop import save_checkpoint
from hagi.train.optim import build_optimizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "rtx3070_fast.yaml"
DEFAULT_CKPT_DIR = ROOT / "checkpoints" / "rtx3070_fast"
DEFAULT_DATA_DIR = ROOT / "data" / "fineweb_1M"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def resolve_train_path(cfg: dict[str, Any], train_path: Path | None, data_dir: Path) -> Path:
    if train_path is not None:
        return train_path
    data_cfg = cfg.get("data", {})
    configured = data_cfg.get("train_path") or data_cfg.get("path")
    if configured:
        return Path(configured)
    bin_files = sorted(data_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"no memmap .bin files found in {data_dir}")
    return bin_files[0]


def lr_at(step: int, max_steps: int, warmup_steps: int, learning_rate: float, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return learning_rate * min_lr_ratio + coeff * learning_rate * (1.0 - min_lr_ratio)


def autocast_ctx(precision: str, device: str):
    if precision == "fp32" or not device.startswith("cuda"):
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def gpu_util(device: str) -> str:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "n/a"
    try:
        index = torch.device(device).index
        util = torch.cuda.utilization(index if index is not None else 0)
        return f"{util}%"
    except Exception:
        return "n/a"


def maybe_compile(model: HAGI, device: str) -> torch.nn.Module:
    if not device.startswith("cuda") or not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model)  # type: ignore[return-value]
    except Exception as exc:
        print(f"torch.compile skipped: {exc}")
        return model


def main() -> None:
    parser = argparse.ArgumentParser(prog="train_rtx3070_fast")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = config_from_dict(cfg.get("model", {}))
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = HAGI(model_cfg).to(args.device)
    if hasattr(model.cfg, "gradient_checkpointing"):
        model.cfg.gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", True))
    optimizer = build_optimizer(model, train_cfg)
    train_model = maybe_compile(model, args.device)
    train_model.train()

    train_path = resolve_train_path(cfg, args.train_path, args.data_dir)
    seq_len = int(data_cfg.get("max_seq_len", 512))
    batch_size = int(train_cfg.get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = bool(data_cfg.get("pin_memory", args.device.startswith("cuda")))
    dataloader = get_memmap_dataloader(
        train_path,
        batch_size=batch_size,
        seq_len=seq_len,
        num_workers=num_workers,
        pin_memory=pin_memory,
        dtype=data_cfg.get("dtype", "uint16"),
    )
    data_iter = iter(dataloader)

    max_steps = int(args.max_steps if args.max_steps is not None else train_cfg.get("max_steps", 20000))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 4))
    warmup_steps = int(train_cfg.get("warmup_steps", 500))
    learning_rate = float(train_cfg.get("learning_rate", 1.0e-3))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    precision = str(train_cfg.get("precision", "fp16"))
    log_interval = int(train_cfg.get("log_interval", 25))
    ckpt_interval = int(train_cfg.get("ckpt_interval", 1000))
    use_scaler = precision == "fp16" and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    tokens_since_log = 0
    last_log_time = start_time
    last_loss = float("nan")

    for step in range(max_steps):
        lr = lr_at(step, max_steps, warmup_steps, learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)
            x = x.to(args.device, non_blocking=pin_memory)
            y = y.to(args.device, non_blocking=pin_memory)
            with autocast_ctx(precision, args.device):
                _, loss = train_model(x, targets=y)
                loss = loss / grad_accum_steps
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item()
            tokens_since_log += x.numel()

        if use_scaler:
            scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        last_loss = accum_loss
        if step % log_interval == 0:
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-9)
            tok_per_sec = tokens_since_log / elapsed
            print(
                f"step {step:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | "
                f"tokens/sec {tok_per_sec:.0f} | gpu_util {gpu_util(args.device)}"
            )
            tokens_since_log = 0
            last_log_time = now

        if ckpt_interval > 0 and step > 0 and step % ckpt_interval == 0:
            save_checkpoint(model, optimizer, step, str(args.ckpt_dir))

    save_checkpoint(model, optimizer, max_steps, str(args.ckpt_dir))
    total_tokens = max_steps * grad_accum_steps * batch_size * seq_len
    total_elapsed = max(time.perf_counter() - start_time, 1e-9)
    print(f"final_loss {last_loss:.4f} | avg_tokens/sec {total_tokens / total_elapsed:.0f}")


if __name__ == "__main__":
    main()
