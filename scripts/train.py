from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import os
import platform
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

# Note: expandable_segments not supported on Windows; skip

from hagi.data import (
    MemmapDataset,
    PrefixLMBatch,
    create_prefix_lm_batch,
    get_memmap_dataloader,
    get_mixed_memmap_dataloader,
    get_sft_dataloader,
)
from hagi.data.tokenizer import TokenizerWrapper
from hagi.losses import composite_loss
from hagi.model import HAGI
from hagi.train.checkpoint import save_checkpoint
from hagi.train.config import config_from_dict, config_to_dict
from hagi.train.loop import LoopConfig, train
from hagi.train.optim import build_optimizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "rtx3070.yaml"
DEFAULT_CKPT_DIR = ROOT / "checkpoints" / "rtx3070"
DEFAULT_DATA_DIR = ROOT / "data" / "fineweb_1M"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def detect_mode(cfg: dict[str, Any]) -> str:
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    if "composite_loss" in train_cfg or "ema" in train_cfg or "use_prefix_lm" in train_cfg or "magic_norm_max" in train_cfg:
        return "full"
    if str(data_cfg.get("dataset_mode", "memmap")).lower() == "sft":
        return "full"
    if "prefetch_factor" in data_cfg or "persistent_workers" in data_cfg:
        return "fast"
    return "basic"


def resolve_train_path(cfg: dict[str, Any], train_path_override: Path | None, data_dir: Path | None = None) -> Path:
    if data_dir is None:
        data_dir = train_path_override
        train_path_override = None
    if train_path_override is not None:
        return Path(train_path_override)

    data_cfg = cfg.get("data", {})
    configured = data_cfg.get("train_path") or data_cfg.get("path")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"train path not found: {path}")
        return path
    if data_dir is None:
        raise ValueError("data_dir is required when no train_path override or configured path")
    bin_files = sorted(data_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"no memmap .bin files found in {data_dir}")
    return bin_files[0]


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
    dataset = MemmapDataset(path, block_size=seq_len, dtype=dtype, preload=True)
    if len(dataset) <= 0:
        raise ValueError(f"memmap dataset is too small for seq_len={seq_len}: {path}")

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(len(dataset), (batch_size,), generator=generator).tolist()
        xs, ys = zip(*(dataset[index] for index in indices), strict=True)
        x = torch.tensor(np.array(xs), dtype=torch.long, device=device)
        y = torch.tensor(np.array(ys), dtype=torch.long, device=device)
        return x, y

    return get_batch


def build_basic_batcher(cfg: dict[str, Any], device: str, train_path: Path | None, data_dir: Path, seq_len: int | None):
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
    if resume.is_dir():
        model_path = resume / "model.pt"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        meta_path = resume / "meta.pt"
        if meta_path.exists():
            meta = torch.load(meta_path, map_location=device, weights_only=True)
            return int(meta.get("step", 0))
        return 0
    state = torch.load(resume, map_location=device, weights_only=True)
    if "model" in state:
        model.load_state_dict(state["model"])
        return int(state.get("step", 0))
    model.load_state_dict(state)
    return 0


def lr_at(
    step: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_lr_ratio: float = 0.1,
    schedule: str = "cosine",
    cooldown_frac: float = 0.05,
) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)
    if str(schedule).lower() == "wsd":
        cooldown_start = int(max_steps * (1.0 - cooldown_frac))
        if step >= cooldown_start:
            cd_progress = (step - cooldown_start) / max(1, max_steps - cooldown_start)
            cd_progress = min(1.0, max(0.0, cd_progress))
            return learning_rate * (min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - cd_progress))
        return learning_rate
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return learning_rate * min_lr_ratio + coeff * learning_rate * (1.0 - min_lr_ratio)


def scheduled_weight(step: int, start: float, final: float, warmup_steps: int, mode: str = "linear") -> float:
    progress = min(1.0, step / max(1, warmup_steps))
    if mode == "cosine":
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (final - start) * progress


def autocast_ctx(precision: str, device: str):
    if precision == "fp32" or not device.startswith("cuda"):
        return torch.autocast(device_type="cpu", enabled=False)
    if precision == "manual_fp16":
        # Manual FP16: model is already in float16, no autocast needed
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
    return model


def to_device(batch: Any, device: str, non_blocking: bool) -> Any:
    if isinstance(batch, PrefixLMBatch):
        return PrefixLMBatch(
            tokens=batch.tokens.to(device, non_blocking=non_blocking),
            mask=batch.mask.to(device, non_blocking=non_blocking),
            partition=batch.partition.to(device, non_blocking=non_blocking),
        )
    if isinstance(batch, tuple):
        return tuple(item.to(device, non_blocking=non_blocking) for item in batch)
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


def _shift_collate(batch: list[Any]) -> tuple[Any, Any]:
    array = np.stack([np.asarray(item, dtype=np.int64) for item in batch])
    x = array[:, :-1]
    y = array[:, 1:]
    return torch.as_tensor(x, dtype=torch.long), torch.as_tensor(y, dtype=torch.long)


def resolve_mix_paths(data_cfg: dict[str, Any], data_dir: Path) -> list[tuple[Path, float]]:
    mix_paths = data_cfg.get("mix_paths", [])
    resolved: list[tuple[Path, float]] = []
    for entry in mix_paths:
        path = Path(entry["path"])
        if not path.exists():
            path = data_dir / path
        resolved.append((path, float(entry["weight"])))
    return resolved


def resolve_eval_path(data_cfg: dict[str, Any], data_dir: Path) -> Path | None:
    configured = data_cfg.get("eval_path")
    if not configured:
        return None
    path = Path(configured)
    if not path.is_absolute():
        path = data_dir / path
    if not path.exists():
        raise FileNotFoundError(f"eval path not found: {path}")
    return path


def build_full_dataloader(
    cfg: dict[str, Any],
    train_path: Path | None,
    data_dir: Path,
    use_prefix_lm: bool,
    device: str,
    eval_samples: int = 0,
    dataset_mode: str = "memmap",
) -> tuple[Any, Any | None, int, int, bool]:
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    seq_len = int(data_cfg.get("max_seq_len", 512))
    batch_size = int(train_cfg.get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", device.startswith("cuda")))
    dataset_mode = dataset_mode.lower()
    if dataset_mode == "memmap_packed":
        dataset_mode = "memmap"

    if dataset_mode == "sft":
        tokenizer_name = str(data_cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
        tokenizer = TokenizerWrapper.smollm2(tokenizer_name, use_fast=True)
        dataset_name = str(data_cfg.get("dataset_name", "HuggingFaceTB/smoltalk"))
        local_path = data_cfg.get("local_path")
        train_loader = get_sft_dataloader(
            dataset_name=dataset_name,
            tokenizer=tokenizer.tokenizer,
            max_seq_len=seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            local_path=local_path,
        )
        eval_loader = None
        return train_loader, eval_loader, batch_size, seq_len, pin_memory

    dtype = data_cfg.get("dtype", "uint16")

    def _make_loader(ds: Any, shuffle: bool, drop_last: bool = True) -> Any:
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
        }
        if use_prefix_lm:
            kwargs["collate_fn"] = partial(prefix_lm_collate, seq_len=seq_len)
        else:
            kwargs["collate_fn"] = _shift_collate
        if num_workers > 0:
            kwargs["prefetch_factor"] = 4
            kwargs["persistent_workers"] = True
        return DataLoader(ds, **kwargs)

    mix_paths = resolve_mix_paths(data_cfg, data_dir) if train_path is None else []
    if mix_paths:
        if use_prefix_lm:
            raise ValueError("mix_paths does not support prefix_lm")
        train_loader = get_mixed_memmap_dataloader(
            mix_paths,
            batch_size=batch_size,
            seq_len=seq_len,
            num_workers=num_workers,
            pin_memory=pin_memory,
            dtype=dtype,
            seed=int(train_cfg.get("seed", 0)),
        )
        eval_path = resolve_eval_path(data_cfg, data_dir)
        eval_loader = (
            _make_loader(MemmapDataset(eval_path, seq_len=seq_len, dtype=dtype), shuffle=False)
            if eval_path is not None
            else None
        )
        return train_loader, eval_loader, batch_size, seq_len, pin_memory

    if train_path is None:
        raise ValueError("train_path is required when no mix_paths")
    dataset = MemmapDataset(train_path, seq_len=seq_len, dtype=dtype)
    total = len(dataset)
    if total <= 0:
        raise ValueError(f"dataset is empty: {train_path}")
    if eval_samples > 0:
        eval_samples = min(eval_samples, total // 10)
        train_ds = Subset(cast(Any, dataset), list(range(total - eval_samples)))
        eval_ds = Subset(cast(Any, dataset), list(range(total - eval_samples, total)))
    else:
        train_ds = dataset
        eval_ds = None
    if len(train_ds) < batch_size:
        raise ValueError(f"train dataset size {len(train_ds)} < batch_size {batch_size}")

    train_loader = _make_loader(train_ds, shuffle=True, drop_last=True)
    eval_loader = _make_loader(eval_ds, shuffle=False, drop_last=False) if eval_ds is not None else None
    return train_loader, eval_loader, batch_size, seq_len, pin_memory


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
    model_output: Any = None,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if weights is None:
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return loss, {}
    # Reuse pre-computed loss from model forward if available
    precomputed_loss = model_output.get("loss") if isinstance(model_output, dict) else None
    moe_aux_loss = model_output.get("moe_aux_loss") if isinstance(model_output, dict) else None
    num_moe_layers = model_output.get("num_moe_layers") if isinstance(model_output, dict) else None
    if num_moe_layers is None:
        # Estimate: 12 layers total (perception + reasoning + expression)
        num_moe_layers = 12
    if precomputed_loss is not None:
        losses = composite_loss(
            logits,
            targets,
            auxiliary_output=model_output.get("auxiliary_output") if isinstance(model_output, dict) else None,
            model_output=model_output.get("model_output") if isinstance(model_output, dict) else None,
            weights=weights,
            invariant_src=model_output.get("invariant_src") if isinstance(model_output, dict) else None,
            invariant_tgt=model_output.get("invariant_tgt") if isinstance(model_output, dict) else None,
            precomputed_loss=precomputed_loss,
            moe_aux_loss=moe_aux_loss,
            num_moe_layers=num_moe_layers,
        )
    else:
        losses = composite_loss(
            logits,
            targets,
            auxiliary_output=model_output.get("auxiliary_output") if isinstance(model_output, dict) else None,
            model_output=model_output.get("model_output") if isinstance(model_output, dict) else None,
            weights=weights,
            invariant_src=model_output.get("invariant_src") if isinstance(model_output, dict) else None,
            invariant_tgt=model_output.get("invariant_tgt") if isinstance(model_output, dict) else None,
            moe_aux_loss=moe_aux_loss,
            num_moe_layers=num_moe_layers,
        )
    total_loss = losses["L_total"]
    components = {name: value.detach().float() for name, value in losses.items()}
    return total_loss, components


def get_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    device = None
    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad
            if device is None:
                device = grad.device
            total += grad.pow(2).sum().item()
    if total == 0.0:
        return 0.0
    return float(total ** 0.5)


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    eval_loader: Any,
    device: str,
    pin_memory: bool,
    precision: str,
    composite_weights: dict[str, float] | None,
    use_prefix_lm: bool,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_ce = 0.0
    total_tokens = 0
    total_correct = 0
    total_aux = 0.0
    total_iso = 0.0
    num_batches = 0
    for batch, targets in eval_loader:
        batch = to_device(batch, device, pin_memory)
        targets = targets.to(device, non_blocking=pin_memory)
        targets = apply_prefix_mask(targets, batch)
        tokens = batch.tokens if isinstance(batch, PrefixLMBatch) else batch
        with autocast_ctx(precision, device):
            output = model(tokens, training_mode=False)
            logits = unwrap_logits(output)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_ce += ce.item()
            valid = targets != -100
            total_tokens += valid.sum().item()
            preds = logits.argmax(dim=-1)
            total_correct += ((preds == targets) & valid).sum().item()
            if composite_weights is not None:
                _, components = compute_loss(logits, targets, output, composite_weights)
                total_aux += components.get("L_aux", torch.tensor(0.0)).item()
                total_iso += components.get("L_iso", torch.tensor(0.0)).item()
        num_batches += 1
    if was_training:
        model.train()
    if num_batches == 0:
        return {"ppl": float("nan"), "acc": float("nan"), "L_aux": float("nan"), "L_iso": float("nan")}
    return {
        "ppl": math.exp(total_ce / max(1, total_tokens)),
        "acc": 100.0 * total_correct / max(1, total_tokens),
        "L_aux": total_aux / max(1, num_batches),
        "L_iso": total_iso / max(1, num_batches),
    }


def update_ema(model: torch.nn.Module, model_ema: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        params = list(model.parameters())
        ema_params = list(model_ema.parameters())
        if hasattr(torch, '_foreach_mul'):
            ema_device = ema_params[0].device if ema_params else None
            if ema_device == params[0].device:
                param_tensors: list[torch.Tensor] = [p for p in params]
                ema_param_tensors: list[torch.Tensor] = [p for p in ema_params]
                torch._foreach_mul_(ema_param_tensors, decay)
                torch._foreach_add_(ema_param_tensors, [p.detach() for p in param_tensors], alpha=1.0 - decay)
            else:
                for ema_param, param in zip(ema_params, params, strict=True):
                    ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        else:
            for ema_param, param in zip(ema_params, params, strict=True):
                ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        for ema_buffer, buffer in zip(model_ema.buffers(), model.buffers(), strict=True):
            ema_buffer.copy_(buffer)


def magic_norm_clip(model: torch.nn.Module, max_norm: float, blade_count: int = 8) -> float:
    gdr = getattr(model, "gdr", None)
    if gdr is None or max_norm <= 0:
        return 0.0
    first_param = next((p for p in gdr.parameters() if p.grad is not None), None)
    if first_param is None:
        return 0.0
    max_norm_t = first_param.new_full((), max_norm, dtype=torch.float32)
    max_seen_t = first_param.new_zeros((), dtype=torch.float32)
    for param in gdr.parameters():
        grad = param.grad
        if grad is None or grad.ndim == 0 or grad.shape[-1] % blade_count != 0:
            continue
        view = grad.view(*grad.shape[:-1], grad.shape[-1] // blade_count, blade_count)
        norms = view.norm(dim=-1, keepdim=True).clamp_min(1e-4).float()
        local_max = norms.max().detach()
        max_seen_t = torch.maximum(max_seen_t, local_max)
        if local_max > max_norm:
            view.mul_((max_norm_t / norms).clamp(max=1.0).to(dtype=view.dtype))
    return float(max_seen_t.item())


def save_training_checkpoint(
    model: HAGI,
    model_ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    ckpt_dir: Path,
) -> None:
    use_moe = getattr(model.cfg, "use_moe", False)
    if use_moe:
        from hagi.train.checkpoint import save_sharded_checkpoint
        path = ckpt_dir / f"step-{step:08d}"
        save_sharded_checkpoint(
            model,
            optimizer,
            model_ema,
            path,
            step=step,
            config=config_to_dict(model.cfg),
        )
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step-{step:08d}.pt"
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
        "optimizer": optimizer.state_dict(),
        "model_ema": {name: value.detach().cpu() for name, value in model_ema.state_dict().items()},
        "config": config_to_dict(model.cfg),
    }
    torch.save(state, path)


def save_inference_only_checkpoint(
    model: HAGI,
    model_ema: torch.nn.Module | None,
    path: Path,
) -> None:
    """Save only model state_dict and optional EMA state_dict for inference."""
    path.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"model": model.state_dict()}
    if model_ema is not None:
        state["ema"] = {name: value.detach().cpu() for name, value in model_ema.state_dict().items()}
    torch.save(state, path / "inference.pt")


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


def run_dry_profile(model: torch.nn.Module, dataloader: Any, device: str, use_prefix_lm: bool, precision: str) -> None:
    model.eval()
    batch = next(iter(dataloader))
    batch = to_device(batch, device, non_blocking=device.startswith("cuda"))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), autocast_ctx(precision, device):
        if use_prefix_lm:
            inputs, targets = batch
            output = model(inputs.tokens)
            logits = unwrap_logits(output)
            targets = apply_prefix_mask(targets, inputs)
        else:
            x, targets = batch
            output = model(x)
            logits = unwrap_logits(output)
        loss, _ = compute_loss(logits, targets)
    print(f"dry_run_loss {float(loss.detach().cpu()):.4f}")
    if device.startswith("cuda"):
        allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
        reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
        print(f"dry_run_peak_vram allocated {allocated:.2f}GB | reserved {reserved:.2f}GB")
    print("dry_run_complete no optimizer step executed")


def run_basic(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
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
    get_batch = build_basic_batcher(cfg, args.device, args.train_path, args.data_dir, args.seq_len)
    loop_cfg = build_loop_config(cfg, args.ckpt_dir, args.max_steps)
    final_loss = train(model, optimizer, get_batch, loop_cfg, device=args.device)
    save_checkpoint(model, optimizer, loop_cfg.max_steps, str(args.ckpt_dir))
    print(f"final_loss {final_loss:.4f}")


def run_fast(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    model_cfg = config_from_dict(cfg.get("model", {}))
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision('high')
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
    num_workers = int(data_cfg.get("num_workers", 4))
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
    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be > 0, got {grad_accum_steps}")
    warmup_steps = int(train_cfg.get("warmup_steps", 500))
    learning_rate = float(train_cfg.get("learning_rate", train_cfg.get("adamw_lr", 1.0e-3)))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    precision = str(train_cfg.get("precision", "fp16"))
    log_interval = int(train_cfg.get("log_interval", 25))
    ckpt_interval = int(train_cfg.get("ckpt_interval", 1000))
    use_scaler = precision == "fp16" and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    if precision == "manual_fp16" and args.device.startswith("cuda"):
        model = model.half()
        print("Using manual FP16: model converted to float16, no autocast")

    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    tokens_since_log = 0
    last_log_time = start_time
    last_loss = float("nan")

    for step in range(max_steps):
        lr = lr_at(step, max_steps, warmup_steps, learning_rate)
        ratio = lr / max(learning_rate, 1e-12)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * ratio

        optimizer.zero_grad(set_to_none=True)
        accum_loss_tensor = None
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
            accum_loss_tensor = loss.detach() if accum_loss_tensor is None else accum_loss_tensor + loss.detach()
            tokens_since_log += x.numel()
            del loss

        if use_scaler:
            scaler.unscale_(optimizer)  # type: ignore[arg-type]
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if use_scaler:
            scaler.step(optimizer)  # type: ignore[arg-type]
            scaler.update()
        else:
            optimizer.step()  # type: ignore[arg-type]

        last_loss = float(accum_loss_tensor.cpu().item()) if accum_loss_tensor is not None else float("nan")
        if log_interval > 0 and step % log_interval == 0:
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-9)
            tok_per_sec = tokens_since_log / elapsed
            print(
                f"step {step:6d} | loss {last_loss:.4f} | lr {lr:.2e} | "
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


def run_full(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    model_cfg = config_from_dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("training", {}))
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate

    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    start_step = 0
    if args.resume is not None and args.resume.exists():
        from hagi.train.loop import load_checkpoint
        model, start_step, _ = load_checkpoint(str(args.resume), args.device)
        model_cfg = model.cfg
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        model = HAGI(model_cfg).to(args.device)
    if hasattr(model.cfg, "gradient_checkpointing"):
        model.cfg.gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", model.cfg.gradient_checkpointing))
    if args.resume is not None and args.resume.exists():
        model_ema = copy.deepcopy(model).to(args.device)
    else:
        model_ema = copy.deepcopy(model).to(args.device)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad_(False)
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
    w_moe_start = float(train_cfg.get("w_moe_start", 0.0))
    w_moe_final = float(train_cfg.get("w_moe_final", composite_weights.get("w_moe", 0.1) if composite_weights else 0.1))
    moe_warmup_steps = int(train_cfg.get("moe_warmup_steps", train_cfg.get("moe_warmup", 2000)))
    loss_warmup_mode = str(train_cfg.get("loss_warmup_mode", "linear"))
    ema_cfg = train_cfg.get("ema", {})
    ema_decay = float(ema_cfg.get("decay", train_cfg.get("ema_decay", 0.999)))
    ema_start_step = int(ema_cfg.get("start_step", train_cfg.get("ema_start_step", 1000)))
    optimizer_kind = str(train_cfg.get("optimizer", "adamw")).lower()
    magic_norm_max = float(train_cfg.get("magic_norm_max", 1.0))
    data_cfg = cfg.get("data", {})
    dataset_mode = args.dataset_mode or str(data_cfg.get("dataset_mode", "memmap"))
    train_path = (
        resolve_train_path(cfg, args.train_path, args.data_dir)
        if args.train_path is not None or not resolve_mix_paths(data_cfg, args.data_dir)
        else None
    )
    eval_interval = int(train_cfg.get("eval_interval", 0))
    eval_samples = int(train_cfg.get("eval_samples", 500))
    dataloader, eval_loader, batch_size, seq_len, pin_memory = build_full_dataloader(
        cfg, train_path, args.data_dir, use_prefix_lm, args.device, eval_samples=eval_samples, dataset_mode=dataset_mode,
    )
    data_iter = iter(dataloader)

    max_steps = int(args.max_steps if args.max_steps is not None else train_cfg.get("max_steps", 50000))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 4))
    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be > 0, got {grad_accum_steps}")
    warmup_steps = int(train_cfg.get("warmup_steps", 500))
    learning_rate = float(train_cfg.get("learning_rate", train_cfg.get("adamw_lr", 5.0e-4)))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    precision = str(train_cfg.get("precision", "fp16"))
    log_interval = int(train_cfg.get("log_interval", 25))
    ckpt_interval = int(train_cfg.get("ckpt_interval", 1000))
    use_scaler = precision == "fp16" and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    if precision == "manual_fp16" and args.device.startswith("cuda"):
        model = model.half()
        model_ema = model_ema.half()
        print("Using manual FP16: model converted to float16, no autocast")

    print_model_summary(model, model_cfg, args.device, use_prefix_lm, composite_weights is not None)
    if args.dry_run:
        run_dry_profile(model, dataloader, args.device, use_prefix_lm, precision)
        return

    optimizer = build_optimizer(model, train_cfg)
    if args.resume is not None and args.resume.exists():
        if args.resume.is_dir():
            optimizer_path = args.resume / "optimizer.pt"
            if optimizer_path.exists():
                try:
                    optimizer.load_state_dict(torch.load(optimizer_path, map_location=args.device, weights_only=True))
                    print("loaded optimizer state")
                except Exception as exc:
                    print(f"optimizer state mismatch, starting fresh: {exc}")
            ema_path = args.resume / "ema.pt"
            if ema_path.exists():
                try:
                    model_ema.load_state_dict(torch.load(ema_path, map_location=args.device, weights_only=True))
                    print("loaded EMA state")
                except Exception as exc:
                    print(f"EMA state mismatch, starting fresh: {exc}")
        else:
            state = torch.load(args.resume, map_location=args.device, weights_only=True)
            if "optimizer" in state:
                try:
                    optimizer.load_state_dict(state["optimizer"])
                    print("loaded optimizer state")
                except Exception as exc:
                    print(f"optimizer state mismatch, starting fresh: {exc}")
            if "model_ema" in state:
                try:
                    model_ema.load_state_dict(state["model_ema"])
                    print("loaded EMA state")
                except Exception as exc:
                    print(f"EMA state mismatch, starting fresh: {exc}")

    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    tokens_since_log = 0
    last_log_time = start_time
    last_loss = float("nan")
    last_components: dict[str, float] = {}
    accum_loss_tensor: torch.Tensor | None = None

    for step in range(start_step, max_steps):
        if optimizer_kind == "schedule-free-adamw":
            lr = learning_rate
        else:
            lr = lr_at(
                step,
                max_steps,
                warmup_steps,
                learning_rate,
                min_lr_ratio,
                schedule=str(train_cfg.get("schedule", "cosine")),
                cooldown_frac=float(train_cfg.get("cooldown_frac", 0.05)),
            )
        ratio = lr / max(learning_rate, 1e-12)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * ratio
        effective_weights = None
        if composite_weights is not None:
            effective_weights = dict(composite_weights)
            effective_weights["w_aux"] = scheduled_weight(step, w_aux_start, w_aux_final, aux_warmup_steps, loss_warmup_mode)
            effective_weights["w_iso"] = scheduled_weight(step, w_iso_start, w_iso_final, iso_warmup_steps, loss_warmup_mode)
            effective_weights["w_moe"] = scheduled_weight(step, w_moe_start, w_moe_final, moe_warmup_steps, loss_warmup_mode)

        optimizer.zero_grad(set_to_none=True)
        accum_loss_tensor = None
        accum_components: dict[str, torch.Tensor] = {}
        need_components = log_interval > 0 and step % log_interval == 0
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
                output = train_model(tokens, targets=targets, training_mode=effective_weights is not None)
                if not train_cfg.get("use_gdr_aux", False):
                    if isinstance(output, dict):
                        output.pop("auxiliary_output", None)
                logits = unwrap_logits(output)
                loss, components = compute_loss(logits, targets, output, effective_weights)
                raw_loss = loss.detach().float()
                loss = loss / grad_accum_steps
            if not torch.isfinite(loss).all():
                if log_interval > 0 and step % log_interval == 0:
                    print(f"WARNING: non-finite loss at step {step}; skipping accum step")
                del output, loss, logits
                continue
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss_tensor = raw_loss if accum_loss_tensor is None else accum_loss_tensor + raw_loss
            if components and need_components:
                for name, value in components.items():
                    accum_components[name] = accum_components.get(name, torch.tensor(0.0, device=value.device, dtype=value.dtype)) + value
            tokens_since_log += tokens.numel()
            del output, loss, logits

        if use_scaler:
            scaler.unscale_(optimizer)  # type: ignore[arg-type]

        if grad_clip > 0:
            full_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if log_interval > 0 and step % log_interval == 0:
                full_grad_norm_val = float(full_grad_norm.item())
                if not math.isfinite(full_grad_norm_val) or full_grad_norm_val > 100.0 or (0.0 < full_grad_norm_val < 1e-6):
                    print(f"WARNING: extreme grad_norm {full_grad_norm_val:.2e} at step {step}")
            else:
                full_grad_norm_val = 0.0
        else:
            full_grad_norm_val = get_grad_norm(model)
            if not math.isfinite(full_grad_norm_val) or full_grad_norm_val > 100.0 or (0.0 < full_grad_norm_val < 1e-6):
                print(f"WARNING: extreme grad_norm {full_grad_norm_val:.2e} at step {step}")

        magic_norm_max_grad = magic_norm_clip(model, magic_norm_max)
        if use_scaler:
            scaler.step(optimizer)  # type: ignore[arg-type]
            scaler.update()
        else:
            optimizer.step()  # type: ignore[arg-type]
        if step >= ema_start_step:
            update_ema(model, model_ema, ema_decay)

        last_loss = float("nan")
        last_components: dict[str, float] = {}
        if log_interval > 0 and step % log_interval == 0:
            last_loss = float((accum_loss_tensor / grad_accum_steps).cpu().item()) if accum_loss_tensor is not None else float("nan")
            if accum_components:
                last_components = {name: (value / grad_accum_steps).item() for name, value in accum_components.items()}
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-9)
            tok_per_sec = tokens_since_log / elapsed
            component_text = ""
            if last_components:
                component_text = " | " + " | ".join(f"{name} {value:.4f}" for name, value in last_components.items())
            weight_text = ""
            if effective_weights is not None:
                weight_text = f" | w_aux {effective_weights['w_aux']:.4f} | w_iso {effective_weights['w_iso']:.4f}"
            eval_model_tag = "ema" if step >= ema_start_step else "model"
            mem_text = ""
            if args.device.startswith("cuda"):
                allocated = torch.cuda.memory_allocated(args.device) / (1024**3)
                reserved = torch.cuda.memory_reserved(args.device) / (1024**3)
                mem_text = f" | mem_allocated {allocated:.2f}GB | mem_reserved {reserved:.2f}GB"
            print(
                f"step {step:6d} | loss {last_loss:.4f}{component_text} | lr {lr:.2e}{weight_text} | "
                f"ema_decay {ema_decay:.4f} | eval_model {eval_model_tag} | grad_norm {full_grad_norm_val:.2e} | "
                f"magic_norm_max_grad {magic_norm_max_grad:.4f} | tokens/sec {tok_per_sec:.0f} | gpu_util {gpu_util(args.device)}{mem_text}"
            )
            tokens_since_log = 0
            last_log_time = now

        if eval_interval > 0 and eval_loader is not None and step > 0 and step % eval_interval == 0:
            eval_model = model_ema if step >= ema_start_step else model
            metrics = run_eval(
                eval_model,
                eval_loader,
                args.device,
                pin_memory,
                precision,
                composite_weights,
                use_prefix_lm,
            )
            eval_tag = "ema" if step >= ema_start_step else "model"
            print(
                f"eval | step {step:6d} | ppl {metrics['ppl']:.2f} | acc {metrics['acc']:.1f}% | "
                f"L_aux {metrics['L_aux']:.4f} | L_iso {metrics['L_iso']:.4f} | {eval_tag}"
            )

        if ckpt_interval > 0 and step > 0 and step % ckpt_interval == 0:
            save_training_checkpoint(model, model_ema, optimizer, step, args.ckpt_dir)  # type: ignore[arg-type]

    save_training_checkpoint(model, model_ema, optimizer, max_steps, args.ckpt_dir)  # type: ignore[arg-type]
    total_tokens = (max_steps - start_step) * grad_accum_steps * batch_size * seq_len
    total_elapsed = max(time.perf_counter() - start_time, 1e-9)
    final_loss_val = float((accum_loss_tensor / grad_accum_steps).cpu().item()) if accum_loss_tensor is not None else float("nan")
    print(f"final_loss {final_loss_val:.4f} | avg_tokens/sec {total_tokens / total_elapsed:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="build model and one batch, report memory, then exit before optimizer step")
    parser.add_argument("--dataset-mode", choices=["memmap", "memmap_packed", "sft"], default=None, help="dataset loading mode (overrides config)")
    parser.add_argument("--mode", choices=["auto", "basic", "fast", "full"], default="auto", help="training mode (auto-detected from config)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    mode = args.mode if args.mode != "auto" else detect_mode(cfg)
    if args.dry_run and mode != "full":
        raise ValueError("--dry-run is only supported in full mode")

    if mode == "basic":
        run_basic(args, cfg)
    elif mode == "fast":
        run_fast(args, cfg)
    elif mode == "full":
        run_full(args, cfg)
    else:
        raise ValueError(f"unknown training mode: {mode}")


if __name__ == "__main__":
    main()
