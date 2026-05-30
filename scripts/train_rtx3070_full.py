from __future__ import annotations

from functools import partial

import argparse
import importlib.util
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from hagi.data import MemmapDataset, PrefixLMBatch, create_prefix_lm_batch, get_memmap_dataloader
from hagi.losses import composite_loss
from hagi.model import HAGI
from hagi.train.checkpoint import save_checkpoint
from hagi.train.config import config_from_dict
from hagi.train.optim import build_optimizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "rtx3070_full.yaml"
DEFAULT_CKPT_DIR = ROOT / "checkpoints" / "rtx3070_full"
DEFAULT_DATA_DIR = ROOT / "data" / "fineweb_1M"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def resolve_train_path(cfg: dict[str, Any], data_dir: Path) -> Path:
    data_cfg = cfg.get("data", {})
    configured = data_cfg.get("train_path") or data_cfg.get("path")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
        return path
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
    if platform.system() == "Windows" and importlib.util.find_spec("triton") is None:
        print("torch.compile skipped: triton is not available on Windows")
        return model
    try:
        return torch.compile(model)  # type: ignore[return-value]
    except Exception as exc:
        print(f"torch.compile skipped: {exc}")
        return model


def to_device(batch: Any, device: str, non_blocking: bool) -> Any:
    if isinstance(batch, PrefixLMBatch):
        return PrefixLMBatch(
            tokens=batch.tokens.to(device, non_blocking=non_blocking),
            mask=batch.mask.to(device, non_blocking=non_blocking),
            partition=batch.partition.to(device, non_blocking=non_blocking),
        )
    return batch.to(device, non_blocking=non_blocking)


def prefix_lm_collate(batch: list[Any], seq_len: int) -> tuple[PrefixLMBatch, torch.Tensor]:
    array = np.stack([np.asarray(item, dtype=np.int64) for item in batch])
    tokens = array[:, :-1]
    targets = torch.as_tensor(array[:, 1:], dtype=torch.long)
    prefix_batch = create_prefix_lm_batch(tokens.tolist(), seq_len)
    return prefix_batch, targets


def build_dataloader(
    cfg: dict[str, Any],
    train_path: Path,
    data_dir: Path,
    use_prefix_lm: bool,
    device: str,
) -> tuple[Any, int, int, bool]:
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    seq_len = int(data_cfg.get("max_seq_len", 512))
    batch_size = int(train_cfg.get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = bool(data_cfg.get("pin_memory", device.startswith("cuda")))
    dtype = data_cfg.get("dtype", "uint16")

    if use_prefix_lm:
        dataset = MemmapDataset(train_path, seq_len=seq_len, dtype=dtype)
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": partial(prefix_lm_collate, seq_len=seq_len),
            "drop_last": True,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = 4
            kwargs["persistent_workers"] = True
        return DataLoader(dataset, **kwargs), batch_size, seq_len, pin_memory

    return (
        get_memmap_dataloader(
            train_path,
            batch_size=batch_size,
            seq_len=seq_len,
            num_workers=num_workers,
            pin_memory=pin_memory,
            dtype=dtype,
        ),
        batch_size,
        seq_len,
        pin_memory,
    )


def unwrap_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        return output["logits"]
    raise TypeError("model output must be a tensor, tuple, or dict")


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    model_output: Any,
    weights: dict[str, float] | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if weights is None:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return loss, {}
    losses = composite_loss(
        logits,
        targets,
        auxiliary_output=model_output.get("auxiliary_output") if isinstance(model_output, dict) else None,
        model_output=model_output.get("model_output") if isinstance(model_output, dict) else None,
        weights=weights,
    )
    return losses["L_total"], {name: value.detach().float().item() for name, value in losses.items()}


def print_model_summary(model: HAGI, cfg: Any, device: str, use_prefix_lm: bool, use_composite_loss: bool) -> None:
    params = model.num_parameters() if hasattr(model, "num_parameters") else sum(p.numel() for p in model.parameters())
    if device.startswith("cuda") and torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        vram_text = f"{reserved:.2f}GB reserved / {vram:.2f}GB total"
    else:
        vram_text = "n/a"
    print(
        "model summary | "
        f"params {params:,} | vram {vram_text} | "
        f"use_loop {cfg.use_loop} | hrm {cfg.hrm} | use_gdr {cfg.use_gdr} | "
        f"hdim_full {cfg.hdim_full} | prefix_lm {use_prefix_lm} | composite_loss {use_composite_loss}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="train_rtx3070_full")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = config_from_dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("training", {}))
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate

    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = HAGI(model_cfg).to(args.device)
    if hasattr(model.cfg, "gradient_checkpointing"):
        model.cfg.gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", model.cfg.gradient_checkpointing))
    optimizer = build_optimizer(model, train_cfg)
    train_model = maybe_compile(model, args.device)
    train_model.train()

    use_prefix_lm = bool(train_cfg.get("use_prefix_lm", False))
    composite_cfg = train_cfg.get("composite_loss")
    composite_weights = dict(composite_cfg) if isinstance(composite_cfg, dict) else None
    train_path = resolve_train_path(cfg, args.data_dir)
    dataloader, batch_size, seq_len, pin_memory = build_dataloader(
        cfg, train_path, args.data_dir, use_prefix_lm, args.device
    )
    data_iter = iter(dataloader)

    max_steps = int(args.max_steps if args.max_steps is not None else train_cfg.get("max_steps", 50000))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 4))
    warmup_steps = int(train_cfg.get("warmup_steps", 500))
    learning_rate = float(train_cfg.get("learning_rate", 5.0e-4))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    precision = str(train_cfg.get("precision", "fp16"))
    log_interval = int(train_cfg.get("log_interval", 25))
    ckpt_interval = int(train_cfg.get("ckpt_interval", 1000))
    use_scaler = precision == "fp16" and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    print_model_summary(model, model_cfg, args.device, use_prefix_lm, composite_weights is not None)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    tokens_since_log = 0
    last_log_time = start_time
    last_loss = float("nan")
    last_components: dict[str, float] = {}

    for step in range(max_steps):
        lr = lr_at(step, max_steps, warmup_steps, learning_rate, min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                batch, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch, targets = next(data_iter)

            batch = to_device(batch, args.device, pin_memory)
            targets = targets.to(args.device, non_blocking=pin_memory)
            tokens = batch.tokens if isinstance(batch, PrefixLMBatch) else batch
            with autocast_ctx(precision, args.device):
                try:
                    output = train_model(tokens, training_mode=composite_weights is not None)
                except TypeError:
                    output = train_model(tokens)
                logits = unwrap_logits(output)
                loss, components = compute_loss(logits, targets, output, composite_weights)
                loss = loss / grad_accum_steps
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item()
            if components:
                last_components = components
            tokens_since_log += tokens.numel()

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
            component_text = ""
            if last_components:
                component_text = " | " + " | ".join(f"{name} {value:.4f}" for name, value in last_components.items())
            print(
                f"step {step:6d} | loss {accum_loss:.4f}{component_text} | lr {lr:.2e} | "
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
