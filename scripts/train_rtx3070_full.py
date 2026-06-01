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
from torch.utils.data import DataLoader, Subset

from hagi.data import MemmapDataset, PrefixLMBatch, create_prefix_lm_batch, get_memmap_dataloader, get_sft_dataloader
from hagi.losses import composite_loss
from hagi.model import HAGI
from hagi.train.checkpoint import save_checkpoint
from hagi.data.tokenizer import TokenizerWrapper
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


def scheduled_weight(step: int, start: float, final: float, warmup_steps: int, mode: str = "linear") -> float:
    progress = min(1.0, step / max(1, warmup_steps))
    if mode == "cosine":
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
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
    if platform.system() == "Windows":
        if importlib.util.find_spec("triton") is None:
            print("torch.compile skipped: triton unavailable on Windows (install triton-windows or use WSL)")
            return model
        import shutil
        if shutil.which("cl") is None and shutil.which("cl.exe") is None:
            print("torch.compile skipped: MSVC compiler (cl) not found on Windows")
            return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")  # type: ignore[return-value]
        # Verify compilation works with a dummy forward pass before returning
        with torch.no_grad():
            dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
            compiled(dummy)
        return compiled
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


def _shift_collate(batch: list[Any]) -> tuple[Any, Any]:
    array = np.stack([np.asarray(item, dtype=np.int64) for item in batch])
    x = array[:, :-1]
    y = array[:, 1:]
    return torch.as_tensor(x, dtype=torch.long), torch.as_tensor(y, dtype=torch.long)


def build_dataloader(
    cfg: dict[str, Any],
    train_path: Path,
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
    dataset = MemmapDataset(train_path, seq_len=seq_len, dtype=dtype)
    total = len(dataset)
    if eval_samples > 0:
        eval_samples = min(eval_samples, total // 10)
        train_ds = Subset(dataset, list(range(total - eval_samples)))
        eval_ds = Subset(dataset, list(range(total - eval_samples, total)))
    else:
        train_ds = dataset
        eval_ds = None

    def _make_loader(ds: Any, shuffle: bool) -> Any:
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": True,
        }
        if use_prefix_lm:
            kwargs["collate_fn"] = partial(prefix_lm_collate, seq_len=seq_len)
        else:
            kwargs["collate_fn"] = _shift_collate
        if num_workers > 0:
            kwargs["prefetch_factor"] = 4
            kwargs["persistent_workers"] = True
        return DataLoader(ds, **kwargs)

    train_loader = _make_loader(train_ds, shuffle=True)
    eval_loader = _make_loader(eval_ds, shuffle=False) if eval_ds is not None else None
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


def get_grad_norm(model: torch.nn.Module) -> float:
    """Compute the L2 norm of gradients across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.float().norm(2).item()
            total_norm += param_norm * param_norm
    return math.sqrt(total_norm)


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
            output = model(tokens, training_mode=True)
            logits = unwrap_logits(output)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
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
                total_aux += components.get("L_aux", 0.0)
                total_iso += components.get("L_iso", 0.0)
        num_batches += 1
    if was_training:
        model.train()
    return {
        "ppl": math.exp(total_ce / max(1, total_tokens)),
        "acc": 100.0 * total_correct / max(1, total_tokens),
        "L_aux": total_aux / max(1, num_batches),
        "L_iso": total_iso / max(1, num_batches),
    }


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
    parser.add_argument("--dataset-mode", choices=["memmap", "sft"], default=None, help="dataset loading mode (overrides config)")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = config_from_dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("training", {}))
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate

    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    start_step = 0
    if args.resume is not None and args.resume.exists():
        from hagi.train.loop import load_checkpoint
        model, start_step = load_checkpoint(str(args.resume), args.device)
        model.to(args.device)
        model_cfg = model.cfg
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        model = HAGI(model_cfg).to(args.device)
    if hasattr(model.cfg, "gradient_checkpointing"):
        model.cfg.gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", model.cfg.gradient_checkpointing))
    if args.resume is not None and args.resume.exists():
        # Rebuild EMA from loaded model state
        model_ema = copy.deepcopy(model).to(args.device)
    else:
        model_ema = copy.deepcopy(model).to(args.device)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad_(False)
    optimizer = build_optimizer(model, train_cfg)
    if args.resume is not None and args.resume.exists():
        # Try to load optimizer state from checkpoint if present
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
    loss_warmup_mode = str(train_cfg.get("loss_warmup_mode", "linear"))
    ema_cfg = train_cfg.get("ema", {})
    ema_decay = float(ema_cfg.get("decay", train_cfg.get("ema_decay", 0.999)))
    ema_start_step = int(ema_cfg.get("start_step", train_cfg.get("ema_start_step", 1000)))
    optimizer_kind = str(train_cfg.get("optimizer", "adamw")).lower()
    magic_norm_max = float(train_cfg.get("magic_norm_max", 1.0))
    data_cfg = cfg.get("data", {})
    dataset_mode = args.dataset_mode or str(data_cfg.get("dataset_mode", "memmap"))
    train_path = resolve_train_path(cfg, args.data_dir)
    eval_interval = int(train_cfg.get("eval_interval", 0))
    eval_samples = int(train_cfg.get("eval_samples", 500))
    dataloader, eval_loader, batch_size, seq_len, pin_memory = build_dataloader(
        cfg, train_path, args.data_dir, use_prefix_lm, args.device, eval_samples=eval_samples, dataset_mode=dataset_mode,
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

    for step in range(start_step, max_steps):
        if optimizer_kind == "schedule-free-adamw":
            lr = learning_rate
        else:
            lr = lr_at(step, max_steps, warmup_steps, learning_rate, min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        effective_weights = None
        if composite_weights is not None:
            effective_weights = dict(composite_weights)
            effective_weights["w_aux"] = scheduled_weight(step, w_aux_start, w_aux_final, aux_warmup_steps, loss_warmup_mode)
            effective_weights["w_iso"] = scheduled_weight(step, w_iso_start, w_iso_final, iso_warmup_steps, loss_warmup_mode)

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

        full_grad_norm = get_grad_norm(model)
        if full_grad_norm > 100.0 or (0.0 < full_grad_norm < 1e-6):
            print(f"WARNING: extreme grad_norm {full_grad_norm:.2e} at step {step}")

        if grad_clip > 0 and full_grad_norm >= grad_clip / 10.0:
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
            eval_model_tag = "ema" if step >= ema_start_step else "model"
            print(
                f"step {step:6d} | loss {last_loss:.4f}{component_text} | lr {lr:.2e}{weight_text} | "
                f"ema_decay {ema_decay:.4f} | eval_model {eval_model_tag} | grad_norm {full_grad_norm:.2e} | "
                f"magic_norm_max_grad {magic_norm_max_grad:.4f} | tokens/sec {tok_per_sec:.0f} | gpu_util {gpu_util(args.device)}"
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
            save_training_checkpoint(model, model_ema, optimizer, step, args.ckpt_dir)

    save_training_checkpoint(model, model_ema, optimizer, max_steps, args.ckpt_dir)
    total_tokens = max_steps * grad_accum_steps * batch_size * seq_len
    total_elapsed = max(time.perf_counter() - start_time, 1e-9)
    print(f"final_loss {last_loss:.4f} | avg_tokens/sec {total_tokens / total_elapsed:.0f}")


if __name__ == "__main__":
    main()
