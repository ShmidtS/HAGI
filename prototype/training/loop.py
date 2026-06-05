"""Core training loop (nanoGPT-adapted, data-source-agnostic).

Wraps the existing HAGI model. Provides: bf16/fp16 autocast, gradient
accumulation, cosine LR schedule with warmup, gradient clipping, periodic
eval + checkpointing. The data source is any zero-arg `get_batch()` returning
(x, y) tensors, so toy data (overfit test) and memmap shards (real training)
share the same loop.
"""

from __future__ import annotations

import math
import time
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
    start_step: int = 0,
    session_steps: int | None = None,
    on_checkpoint: Callable[[str], None] | None = None,
):
    """Run the training loop. Returns the final training loss.

    optimizer: torch.optim.Optimizer or CombinedOptimizer (Muon+AdamW).
    on_log: optional callback receiving a metrics dict each log step.
    start_step: resume from this step (LR schedule + intervals are absolute).
    session_steps: if set, stop after this many steps this run (checkpoint-gated
        local training). A checkpoint is always written at session end so the next
        `--resume auto` continues exactly where this one stopped.
    on_checkpoint: optional callback receiving each saved checkpoint path (used to
        mirror checkpoints to HF Hub for cross-session/cross-machine resume).
    """
    model.to(device)
    model.train()
    use_scaler = cfg.precision == "fp16" and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    end = cfg.max_steps if session_steps is None else min(cfg.max_steps, start_step + session_steps)
    last_loss = float("nan")
    ran = False
    t_log = time.time()
    tokens_since_log = 0
    for step in range(start_step, end):
        ran = True
        lr = _lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch()
            tokens_since_log += x.numel()
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
            dt = time.time() - t_log
            tps = tokens_since_log / dt if dt > 0 else 0.0
            metrics = {"step": step, "loss": accum_loss, "lr": lr, "tok_per_s": tps}
            if on_log:
                on_log(metrics)
            else:
                print(f"step {step:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | {tps:,.0f} tok/s")
            t_log = time.time()
            tokens_since_log = 0

        if eval_get_batch is not None and cfg.eval_interval > 0 and step > 0 \
                and step % cfg.eval_interval == 0:
            val = estimate_loss(model, eval_get_batch, cfg.eval_iters, device, cfg.precision)
            print(f"step {step:6d} | val_loss {val:.4f}")

        if cfg.ckpt_interval > 0 and step > 0 and step % cfg.ckpt_interval == 0:
            p = save_checkpoint(model, optimizer, step, cfg.ckpt_dir)
            if on_checkpoint:
                on_checkpoint(p)

    # Always checkpoint at session end (labelled `end` = next resume point), so a
    # gated session boundary is exactly resumable without redoing a step.
    if ran:
        p = save_checkpoint(model, optimizer, end, cfg.ckpt_dir)
        if on_checkpoint:
            on_checkpoint(p)
    return last_loss


def _unwrap(model):
    """Return the underlying HAGI module, unwrapping a torch.compile wrapper.

    torch.compile returns an OptimizedModule whose state_dict keys are prefixed
    with `_orig_mod.`; saving/loading must use the original module to keep keys
    stable across compiled and uncompiled runs.
    """
    return getattr(model, "_orig_mod", model)


def save_checkpoint(model: HAGI, optimizer, step: int, ckpt_dir: str) -> str:
    """Write a checkpoint. Config is stored as a plain dict (not a pickled
    dataclass) so the file loads under torch's default weights_only=True.
    Optimizer state is included when provided, enabling exact resume. Returns the
    checkpoint path."""
    from prototype.training.config import config_to_dict

    base = _unwrap(model)
    out = Path(ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"step-{step:08d}.pt"
    payload = {"model": base.state_dict(), "step": step, "config": config_to_dict(base.cfg)}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    print(f"checkpoint -> {path}")
    return str(path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[HAGI, int]:
    """Rebuild a HAGI model from a checkpoint (model only — for eval/inference).
    Loads under weights_only=True (config is a plain dict, weights are tensors —
    no arbitrary unpickling)."""
    from prototype.training.config import config_from_dict

    state = torch.load(path, map_location=device, weights_only=True)
    cfg = config_from_dict(state["config"])
    model = HAGI(cfg)
    model.load_state_dict(state["model"])
    model.to(device)
    return model, int(state.get("step", 0))


def latest_checkpoint(ckpt_dir: str) -> Path | None:
    """Newest `step-*.pt` in ckpt_dir, or None if there are none."""
    cks = sorted(Path(ckpt_dir).glob("step-*.pt"))
    return cks[-1] if cks else None


def resume_into(model: HAGI, optimizer, path: str, device: str = "cpu") -> int:
    """Load weights (and optimizer state, if present) into an existing model +
    optimizer. Returns the step to resume from. For free-tier cloud where 12h
    sessions die mid-run, this restores both so training continues seamlessly."""
    base = _unwrap(model)
    state = torch.load(path, map_location=device, weights_only=True)
    base.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("step", 0))
