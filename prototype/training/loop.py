"""Core training loop (nanoGPT-adapted, data-source-agnostic).

Wraps the existing HAGI model. Provides: bf16/fp16 autocast, gradient
accumulation, cosine LR schedule with warmup, gradient clipping, periodic
eval + checkpointing. The data source is any zero-arg `get_batch()` returning
(x, y) tensors, so toy data (overfit test) and memmap shards (real training)
share the same loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from prototype.model.hagi import HAGI


@dataclass
class LoopConfig:
    max_steps: int = 50000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    precision: str = "bf16"        # "bf16" | "fp16" | "fp32"
    eval_interval: int = 2000
    eval_iters: int = 50
    ckpt_interval: int = 5000
    ckpt_dir: str = "checkpoints"
    log_interval: int = 50


def _lr_at(step: int, cfg: LoopConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    return min_lr + coeff * (cfg.learning_rate - min_lr)


def _autocast_ctx(precision: str, device: str):
    if precision == "fp32" or not device.startswith("cuda"):
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def estimate_loss(model: HAGI, get_batch: Callable, iters: int, device: str, precision: str) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch()
        with _autocast_ctx(precision, device):
            _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(
    model: HAGI,
    optimizer,
    get_batch: Callable,
    cfg: LoopConfig,
    device: str = "cpu",
    eval_get_batch: Callable | None = None,
    on_log: Callable[[dict], None] | None = None,
):
    """Run the training loop. Returns the final training loss.

    optimizer: torch.optim.Optimizer or CombinedOptimizer (Muon+AdamW).
    on_log: optional callback receiving a metrics dict each log step.
    """
    model.to(device)
    model.train()
    use_scaler = cfg.precision == "fp16" and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    last_loss = float("nan")
    for step in range(cfg.max_steps):
        lr = _lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch()
            with _autocast_ctx(cfg.precision, device):
                _, loss = model(x, targets=y)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward() if use_scaler else loss.backward()
            accum_loss += loss.item()

        if use_scaler:
            scaler.unscale_(optimizer)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        last_loss = accum_loss
        if step % cfg.log_interval == 0:
            metrics = {"step": step, "loss": accum_loss, "lr": lr}
            if on_log:
                on_log(metrics)
            else:
                print(f"step {step:6d} | loss {accum_loss:.4f} | lr {lr:.2e}")

        if eval_get_batch is not None and cfg.eval_interval > 0 and step > 0 \
                and step % cfg.eval_interval == 0:
            val = estimate_loss(model, eval_get_batch, cfg.eval_iters, device, cfg.precision)
            print(f"step {step:6d} | val_loss {val:.4f}")

        if cfg.ckpt_interval > 0 and step > 0 and step % cfg.ckpt_interval == 0:
            save_checkpoint(model, optimizer, step, cfg.ckpt_dir)

    return last_loss


def save_checkpoint(model: HAGI, optimizer, step: int, ckpt_dir: str):
    out = Path(ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"step-{step:08d}.pt"
    torch.save(
        {"model": model.state_dict(), "step": step, "config": model.cfg},
        path,
    )
    print(f"checkpoint -> {path}")
