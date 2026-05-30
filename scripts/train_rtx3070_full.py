from __future__ import annotations

from functools import partial

import argparse
import copy
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
from hagi.train.config import config_from_dict, config_to_dict
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


def scheduled_weight(step: int, start: float, final: float, warmup_steps: int) -> float:
    progress = min(1.0, step / max(1, warmup_steps))
    return start + (final - start) * progress


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


def apply_prefix_mask(targets: torch.Tensor, batch: Any) -> torch.Tensor:
    if not isinstance(batch, PrefixLMBatch):
        return targets
    masked = targets.clone()
    positions = torch.arange(masked.size(1), device=masked.device).unsqueeze(0)
    masked[positions < batch.partition.unsqueeze(1)] = -100
    return masked


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
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return loss, {}
    losses = composite_loss(
        logits,
        targets,
        auxiliary_output=model_output.get("auxiliary_output") if isinstance(model_output, dict) else None,
        model_output=model_output.get("model_output") if isinstance(model_output, dict) else None,
        weights=weights,
        invariant_src=model_output.get("invariant_src") if isinstance(model_output, dict) else None,
        invariant_tgt=model_output.get("invariant_tgt") if isinstance(model_output, dict) else None,
    )
    return losses["L_total"], {name: value.detach().float().item() for name, value in losses.items()}


def update_ema(model: torch.nn.Module, model_ema: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(model_ema.parameters(), model.parameters(), strict=True):
            ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        for ema_buffer, buffer in zip(model_ema.buffers(), model.buffers(), strict=True):
            ema_buffer.copy_(buffer)


def magic_norm_clip(model: torch.nn.Module, max_norm: float, blade_count: int = 8) -> float:
    gdr = getattr(model, "gdr", None)
    if gdr is None or max_norm <= 0:
        return 0.0
    max_seen = 0.0
    for param in gdr.parameters():
        grad = param.grad
        if grad is None or grad.ndim == 0 or grad.shape[-1] % blade_count != 0:
            continue
        view = grad.view(*grad.shape[:-1], grad.shape[-1] // blade_count, blade_count)
        norms = view.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        max_seen = max(max_seen, float(norms.max().item()))
        view.mul_((max_norm / norms).clamp(max=1.0).to(dtype=view.dtype))
    return max_seen


def save_training_checkpoint(
    model: HAGI,
    model_ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    ckpt_dir: Path,
) -> None:
    save_checkpoint(model, optimizer, step, str(ckpt_dir))
    path = ckpt_dir / f"step-{step:08d}.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["model_ema"] = {name: value.detach().cpu() for name, value in model_ema.state_dict().items()}
    state["optimizer"] = optimizer.state_dict()
    state["config"] = config_to_dict(model.cfg)
    torch.save(state, path)


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
    model_ema = copy.deepcopy(model).to(args.device)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad_(False)
    optimizer = build_optimizer(model, train_cfg)
    train_model = maybe_compile(model, args.device)
    train_model.train()

    use_prefix_lm = bool(train_cfg.get("use_prefix_lm", False))
    composite_cfg = train_cfg.get("composite_loss")
    composite_weights = dict(composite_cfg) if isinstance(composite_cfg, dict) else None
    w_aux_start = float(train_cfg.get("w_aux_start", 0.0))
    w_aux_final = float(train_cfg.get("w_aux_final", composite_weights.get("w_aux", 0.1) if composite_weights else 0.1))
    aux_warmup_steps = int(train_cfg.get("aux_warmup_steps", train_cfg.get("aux_warmup", 2000)))
    w_iso_start = float(train_cfg.get("w_iso_start", 0.0))
    w_iso_final = float(train_cfg.get("w_iso_final", composite_weights.get("w_iso", 0.01) if composite_weights else 0.01))
    iso_warmup_steps = int(train_cfg.get("iso_warmup_steps", train_cfg.get("iso_warmup", 5000)))
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    ema_start_step = int(train_cfg.get("ema_start_step", 1000))
    magic_norm_max = float(train_cfg.get("magic_norm_max", 1.0))
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
        effective_weights = None
        if composite_weights is not None:
            effective_weights = dict(composite_weights)
            effective_weights["w_aux"] = scheduled_weight(step, w_aux_start, w_aux_final, aux_warmup_steps)
            effective_weights["w_iso"] = scheduled_weight(step, w_iso_start, w_iso_final, iso_warmup_steps)

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_components: dict[str, float] = {}
        for _ in range(grad_accum_steps):
            try:
                batch, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch, targets = next(data_iter)

            batch = to_device(batch, args.device, pin_memory)
            targets = targets.to(args.device, non_blocking=pin_memory)
            targets = apply_prefix_mask(targets, batch)
            tokens = batch.tokens if isinstance(batch, PrefixLMBatch) else batch
            with autocast_ctx(precision, args.device):
                try:
                    output = train_model(tokens, training_mode=effective_weights is not None)
                except TypeError:
                    output = train_model(tokens)
                logits = unwrap_logits(output)
                loss, components = compute_loss(logits, targets, output, effective_weights)
                raw_loss = loss.detach().float().item()
                loss = loss / grad_accum_steps
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += raw_loss
            if components:
                for name, value in components.items():
                    accum_components[name] = accum_components.get(name, 0.0) + value
            tokens_since_log += tokens.numel()

        if accum_components:
            last_components = {name: value / grad_accum_steps for name, value in accum_components.items()}

        if use_scaler:
            scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        magic_norm_max_grad = magic_norm_clip(model, magic_norm_max)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if step >= ema_start_step:
            update_ema(model, model_ema, ema_decay)

        last_loss = accum_loss / grad_accum_steps
        if step % log_interval == 0:
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-9)
            tok_per_sec = tokens_since_log / elapsed
            component_text = ""
            if last_components:
                component_text = " | " + " | ".join(f"{name} {value:.4f}" for name, value in last_components.items())
            weight_text = ""
            if effective_weights is not None:
                weight_text = f" | w_aux {effective_weights['w_aux']:.4f} | w_iso {effective_weights['w_iso']:.4f}"
            eval_model = "ema" if step >= ema_start_step else "model"
            print(
                f"step {step:6d} | loss {last_loss:.4f}{component_text} | lr {lr:.2e}{weight_text} | "
                f"ema_decay {ema_decay:.4f} | eval_model {eval_model} | magic_norm_max_grad {magic_norm_max_grad:.4f} | "
                f"tokens/sec {tok_per_sec:.0f} | gpu_util {gpu_util(args.device)}"
            )
            tokens_since_log = 0
            last_log_time = now

        if ckpt_interval > 0 and step > 0 and step % ckpt_interval == 0:
            save_training_checkpoint(model, model_ema, optimizer, step, args.ckpt_dir)

    save_training_checkpoint(model, model_ema, optimizer, max_steps, args.ckpt_dir)
    total_tokens = max_steps * grad_accum_steps * batch_size * seq_len
    total_elapsed = max(time.perf_counter() - start_time, 1e-9)
    print(f"final_loss {last_loss:.4f} | avg_tokens/sec {total_tokens / total_elapsed:.0f}")


if __name__ == "__main__":
    main()
