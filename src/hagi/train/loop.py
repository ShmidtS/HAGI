"""Core training loop (nanoGPT-adapted, data-source-agnostic).

Wraps the HAGI model. Provides: bf16/fp16 autocast, gradient accumulation,
cosine LR schedule with warmup, gradient clipping, periodic eval +
checkpointing. The data source is any zero-arg `get_batch()` returning (x, y)
tensors, so toy data and memmap shards share the same loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch

from hagi.train.config import config_from_dict, config_to_dict

if TYPE_CHECKING:
    from hagi.model import HAGI


@dataclass
class LoopConfig:
    max_steps: int = 50000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    precision: str = "bf16"
    gradient_checkpointing: bool = False
    eval_interval: int = 2000
    eval_iters: int = 50
    ckpt_interval: int = 5000
    ckpt_dir: str = "checkpoints"
    log_interval: int = 50


# NARS re-applies its resolved HRM control policy (mutating model.hrm
# h_cycles/l_cycles) on this step interval. observe_train_step still runs every
# step to keep truth/budget statistics fresh; only the cycle-mutating apply is
# throttled. See loop body for rationale.
NARS_POLICY_INTERVAL = 200


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
def estimate_loss(
    model: HAGI, get_batch: Callable[..., Any], iters: int, device: str, precision: str
) -> float:
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
    get_batch: Callable[..., Any],
    cfg: LoopConfig,
    device: str = "cpu",
    eval_get_batch: Callable[..., Any] | None = None,
    on_log: Callable[[dict[str, Any]], None] | None = None,
    start_step: int = 0,
    session_steps: int | None = None,
    on_checkpoint: Callable[[str], None] | None = None,
):
    """Run the training loop. Returns the final training loss."""
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    model.to(device)
    model.train()
    if hasattr(model.cfg, "gradient_checkpointing"):
        model.cfg.gradient_checkpointing = cfg.gradient_checkpointing
    # Opt-in whole-model compile (model cfg.compile). Forward passes go through
    # `run_model`; checkpoints keep using `model` so state_dict keys never get
    # the `_orig_mod.` prefix.
    run_model = model
    if getattr(getattr(model, "cfg", None), "compile", False) and device.startswith(
        "cuda"
    ):
        if hasattr(torch, "compile"):
            run_model = torch.compile(model)
    use_scaler = cfg.precision == "fp16" and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # NARS HRM controller setup
    nars_hrm = None
    if hasattr(model, "nars_hrm") and model.nars_hrm is not None:
        nars_hrm = model.nars_hrm

    last_loss = float("nan")
    end = (
        cfg.max_steps
        if session_steps is None
        else min(cfg.max_steps, start_step + session_steps)
    )
    for step in range(start_step, end):
        lr = _lr_at(step, cfg)
        base_lr = max(cfg.learning_rate, 1e-12)
        for group in optimizer.param_groups:
            init_lr = group.get("initial_lr", group["lr"])
            group["lr"] = init_lr * (lr / base_lr)

        optimizer.zero_grad(set_to_none=True)
        accum_loss_tensor = None
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch()
            with _autocast_ctx(cfg.precision, device):
                result = run_model(x, targets=y, training_mode=True)
                loss = result["loss"] if isinstance(result, dict) else result[1]
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward() if use_scaler else loss.backward()
            accum_loss_tensor = (
                loss.detach()
                if accum_loss_tensor is None
                else accum_loss_tensor + loss.detach()
            )
        accum_loss = (
            accum_loss_tensor.item() if accum_loss_tensor is not None else float("nan")
        )

        if use_scaler:
            scaler.unscale_(optimizer)
        grad_norm: float | None = None
        if cfg.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            ).item()

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        last_loss = accum_loss

        # NARS HRM control: observe every step (cheap stat accumulation), but
        # only re-apply the resolved policy (which mutates model.hrm cycles and
        # thereby changes the model's math) on a coarse interval. Per-step
        # cycle churn destabilises training (grad spikes, loss plateau) and
        # makes forward depth unpredictable. cycles: observe=always, apply=200.
        if nars_hrm is not None:
            if grad_norm is None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf")
                ).item()
            nars_hrm.observe_train_step(last_loss, grad_norm)
            if step % NARS_POLICY_INTERVAL == 0:
                policy = nars_hrm.resolve_policy()
                if hasattr(model, "hrm") and model.hrm is not None:
                    nars_hrm.apply_policy(policy, model.hrm)

        if step % cfg.log_interval == 0:
            metrics = {"step": step, "loss": accum_loss, "lr": lr}
            # Log the ACTUAL HRM cycles in effect (config-driven, not the NARS
            # proposed policy — apply_policy no longer mutates cycles).
            if hasattr(model, "hrm") and model.hrm is not None:
                metrics["h_cycles"] = getattr(model.hrm, "h_cycles", None)
                metrics["l_cycles"] = getattr(model.hrm, "l_cycles", None)
            if on_log:
                on_log(metrics)
            else:
                extras = ""
                if hasattr(model, "hrm") and model.hrm is not None:
                    extras = f" | h={getattr(model.hrm, 'h_cycles', '?')} l={getattr(model.hrm, 'l_cycles', '?')}"
                print(f"step {step:6d} | loss {accum_loss:.4f} | lr {lr:.2e}{extras}")

        if (
            eval_get_batch is not None
            and cfg.eval_interval > 0
            and step > 0
            and step % cfg.eval_interval == 0
        ):
            val = estimate_loss(
                model, eval_get_batch, cfg.eval_iters, device, cfg.precision
            )
            print(f"step {step:6d} | val_loss {val:.4f}")

        if cfg.ckpt_interval > 0 and step > 0 and step % cfg.ckpt_interval == 0:
            save_checkpoint(
                model, optimizer, step, cfg.ckpt_dir, on_checkpoint=on_checkpoint
            )

    if session_steps is not None and on_checkpoint is not None:
        save_checkpoint(
            model, optimizer, end, cfg.ckpt_dir, on_checkpoint=on_checkpoint
        )

    return last_loss


def save_checkpoint(
    model: HAGI,
    optimizer,
    step: int,
    ckpt_dir: str,
    ema_state: dict[str, Any] | None = None,
    on_checkpoint: Callable[[str], None] | None = None,
):
    """Write a checkpoint with config, optimizer, and optional EMA state."""
    out = Path(ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"step-{step:08d}.pt"
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
        "config": config_to_dict(model.cfg),
        "optimizer": optimizer.state_dict(),
    }
    if ema_state is not None:
        payload["model_ema"] = ema_state
    # Save MSA registry slots if present
    if hasattr(model, "msa_registry") and model.msa_registry is not None:
        payload["msa_registry"] = model.msa_registry.state_dict()
    # Save NARS adapter states if present
    if hasattr(model, "nars_hrm") and model.nars_hrm is not None:
        payload["nars_hrm"] = model.nars_hrm.state_dict()
    if hasattr(model, "nars_hdim") and model.nars_hdim is not None:
        payload["nars_hdim"] = model.nars_hdim.state_dict()
    if hasattr(model, "nars_msa") and model.nars_msa is not None:
        payload["nars_msa"] = model.nars_msa.state_dict()
    torch.save(payload, path)
    print(f"checkpoint -> {path}")
    if on_checkpoint is not None:
        on_checkpoint(str(path))


def load_checkpoint(
    path: str,
    device: str = "cpu",
    optimizer=None,
    load_ema: bool = False,
    use_ema: bool = False,
) -> tuple[HAGI, int, dict[str, Any] | None]:
    """Rebuild a HAGI model from a checkpoint.

    Args:
        path: checkpoint path
        device: target device
        optimizer: optional optimizer to load state into
        load_ema: whether to return EMA state dict
        use_ema: whether to load EMA weights into the model itself (inference)

    Returns:
        (model, step, ema_state | None)
    """
    from hagi.model import HAGI
    from pathlib import Path

    p = Path(path)
    # Handle sharded checkpoint directories (model.pt, optimizer.pt, ema.pt, meta.pt)
    if p.is_dir() and (p / "model.pt").exists():
        meta = (
            torch.load(p / "meta.pt", map_location=device, weights_only=True)
            if (p / "meta.pt").exists()
            else {}
        )
        cfg = config_from_dict(meta["config"])
        model = HAGI(cfg)
        state_dict = torch.load(p / "model.pt", map_location=device, weights_only=True)
        # Normalize keys saved from a torch.compiled model
        if any(k.startswith("hrm._orig_mod.") for k in state_dict):
            state_dict = {
                k.replace("hrm._orig_mod.", "hrm.", 1): v for k, v in state_dict.items()
            }
        # Strip orphaned fused-projection keys emitted at top level by old state_dict
        for key in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
            state_dict.pop(key, None)
        model.load_state_dict(state_dict)
        model.to(device)
        ema_state = None
        if (use_ema or load_ema) and (p / "ema.pt").exists():
            ema_state = torch.load(p / "ema.pt", map_location=device, weights_only=True)
            if use_ema:
                model.load_state_dict(ema_state)
                ema_state = None
        return model, int(meta.get("step", 0)), ema_state

    state = torch.load(path, map_location=device, weights_only=True)
    # Strip orphaned fused-projection keys emitted at top level by old state_dict
    for key in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
        state.pop(key, None)
    cfg = config_from_dict(state["config"])
    model = HAGI(cfg)

    if state.get("_inference_opt"):
        from hagi.model.inference_opt import (
            fold_rmsnorm_into_weights,
            precompute_rope_tables,
            repack_qkv_for_contiguous,
        )

        max_seq_len = state.get("_rope_max_seq_len", cfg.transformer.max_seq_len)
        fold_rmsnorm_into_weights(model)
        repack_qkv_for_contiguous(model)
        precompute_rope_tables(model, max_seq_len)

    model.load_state_dict(state["model"])
    model.to(device)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    # Load MSA registry if present
    if (
        hasattr(model, "msa_registry")
        and model.msa_registry is not None
        and "msa_registry" in state
    ):
        model.msa_registry.load_state_dict(state["msa_registry"])

    # Load NARS adapter states if present
    if (
        hasattr(model, "nars_hrm")
        and model.nars_hrm is not None
        and "nars_hrm" in state
    ):
        model.nars_hrm.load_state_dict(state["nars_hrm"])
    if (
        hasattr(model, "nars_hdim")
        and model.nars_hdim is not None
        and "nars_hdim" in state
    ):
        model.nars_hdim.load_state_dict(state["nars_hdim"])
    if (
        hasattr(model, "nars_msa")
        and model.nars_msa is not None
        and "nars_msa" in state
    ):
        model.nars_msa.load_state_dict(state["nars_msa"])

    ema_state = state.get("model_ema") if (use_ema or load_ema) else None
    if use_ema and ema_state is not None:
        model.load_state_dict(ema_state)
        ema_state = None
    return model, int(state.get("step", 0)), ema_state
