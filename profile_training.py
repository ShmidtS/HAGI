"""Comprehensive training profiler for HAGI.

Profiles: forward pass, backward pass, optimizer step, data loading,
memory usage (VRAM/RAM), CPU/GPU breakdown per component.

Usage: PYTHONPATH=src python profile_training.py --config configs/rtx3070_canonical.yaml --steps 10
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from hagi.model import HAGI
from hagi.train.config import config_from_dict
from hagi.train.optim import build_optimizer
from hagi.data import MemmapDataset, get_memmap_dataloader


def get_gpu_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,
        "reserved": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
    }


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def format_time_ms(t: float) -> str:
    if t < 0.001:
        return f"{t * 1e6:.1f} µs"
    if t < 1:
        return f"{t * 1e3:.1f} ms"
    return f"{t:.2f} s"


def profile_stage(name: str, fn, device: str, n_warmup: int = 2, n_iter: int = 5):
    """Profile a function: warmup then measure."""
    # Warmup
    for _ in range(n_warmup):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        _ = fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    mem_before = get_gpu_memory()
    reset_peak_memory()

    for _ in range(n_iter):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mem_after = get_gpu_memory()
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n{'='*60}")
    print(f"Stage: {name}")
    print(f"{'='*60}")
    print(f"  Time: avg={format_time_ms(avg_time)}  min={format_time_ms(min_time)}  max={format_time_ms(max_time)}")
    print(f"  GPU mem: allocated={mem_after['allocated']:.2f}GB  reserved={mem_after['reserved']:.2f}GB  peak={mem_after['max_allocated']:.2f}GB")
    print(f"  Delta: +{mem_after['allocated'] - mem_before['allocated']:.2f}GB allocated")

    return result, avg_time


def profile_full_training(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(open(args.config)) if isinstance(args.config, (str, Path)) else {}
    model_cfg = config_from_dict(cfg.get("model", {}))
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    device = args.device
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print("HAGI Training Profiler")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Profile steps: {args.steps}")

    # 1. Model creation
    print("\n" + "=" * 60)
    print("1. MODEL CREATION")
    print("=" * 60)
    reset_peak_memory()
    t0 = time.perf_counter()
    model = HAGI(model_cfg).to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    mem = get_gpu_memory()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Time: {format_time_ms(t1 - t0)}")
    print(f"  Params: {params:,} total, {trainable:,} trainable")
    print(f"  GPU mem: {mem['allocated']:.2f}GB allocated, {mem['reserved']:.2f}GB reserved")

    # 2. Optimizer creation
    print("\n" + "=" * 60)
    print("2. OPTIMIZER CREATION")
    print("=" * 60)
    reset_peak_memory()
    t0 = time.perf_counter()
    optimizer = build_optimizer(model, train_cfg)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    mem = get_gpu_memory()
    print(f"  Time: {format_time_ms(t1 - t0)}")
    print(f"  GPU mem: {mem['allocated']:.2f}GB allocated, {mem['reserved']:.2f}GB reserved")

    # 3. Data loading
    print("\n" + "=" * 60)
    print("3. DATA LOADING")
    print("=" * 60)
    data_dir = Path(args.data_dir)
    train_path = data_dir / "edu.bin"
    seq_len = int(data_cfg.get("max_seq_len", 1024))
    batch_size = int(train_cfg.get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    reset_peak_memory()
    t0 = time.perf_counter()
    dataloader = get_memmap_dataloader(
        train_path,
        batch_size=batch_size,
        seq_len=seq_len,
        num_workers=num_workers,
        pin_memory=pin_memory,
        dtype=data_cfg.get("dtype", "uint16"),
    )
    data_iter = iter(dataloader)
    # Fetch first batch
    x, y = next(data_iter)
    x = x.to(device, non_blocking=pin_memory)
    y = y.to(device, non_blocking=pin_memory)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    mem = get_gpu_memory()
    print(f"  Time: {format_time_ms(t1 - t0)}")
    print(f"  Batch shape: x={tuple(x.shape)}, y={tuple(y.shape)}")
    print(f"  GPU mem: {mem['allocated']:.2f}GB allocated")

    # 4. Forward pass profiling
    print("\n" + "=" * 60)
    print("4. FORWARD PASS")
    print("=" * 60)
    model.train()
    precision = str(train_cfg.get("precision", "fp16"))
    use_scaler = precision == "fp16" and device.startswith("cuda")
    use_autocast = use_scaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if precision == "manual_fp16" and device.startswith("cuda"):
        model = model.half()
        print("Using manual FP16: model converted to float16, no autocast")

    grad_accum = int(train_cfg.get("grad_accum_steps", 8))
    composite_cfg = train_cfg.get("composite_loss")
    use_composite = composite_cfg is not None

    def forward_fn():
        with torch.autocast("cuda", dtype=torch.float16 if use_autocast else torch.float32, enabled=use_autocast):
            output = model(x, targets=y, training_mode=use_composite)
        return output

    _, fwd_time = profile_stage("Forward Pass", forward_fn, device, n_warmup=3, n_iter=5)

    # 5. Backward pass profiling
    print("\n" + "=" * 60)
    print("5. BACKWARD PASS")
    print("=" * 60)

    def backward_fn():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16 if use_autocast else torch.float32, enabled=use_autocast):
            output = model(x, targets=y, training_mode=use_composite)
        if isinstance(output, dict):
            loss = output["loss"] if "loss" in output else output["logits"]
        else:
            loss = output[1] if isinstance(output, tuple) else output
        loss = loss / grad_accum
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss

    _, bwd_time = profile_stage("Backward Pass", backward_fn, device, n_warmup=2, n_iter=5)

    # 6. Optimizer step profiling
    print("\n" + "=" * 60)
    print("6. OPTIMIZER STEP")
    print("=" * 60)

    def optimizer_fn():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16 if use_autocast else torch.float32, enabled=use_autocast):
            output = model(x, targets=y, training_mode=use_composite)
        if isinstance(output, dict):
            loss = output["loss"] if "loss" in output else output["logits"]
        else:
            loss = output[1] if isinstance(output, tuple) else output
        loss = loss / grad_accum
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return loss

    _, opt_time = profile_stage("Full Step (fwd+bwd+opt)", optimizer_fn, device, n_warmup=2, n_iter=5)

    # 7. Component breakdown with torch.profiler
    print("\n" + "=" * 60)
    print("7. COMPONENT BREAKDOWN (torch.profiler)")
    print("=" * 60)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16 if use_autocast else torch.float32, enabled=use_autocast):
            output = model(x, targets=y, training_mode=use_composite)
        if isinstance(output, dict):
            loss = output["loss"] if "loss" in output else output["logits"]
        else:
            loss = output[1] if isinstance(output, tuple) else output
        loss = loss / grad_accum
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    print("\nTop 20 CUDA operations by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\nTop 20 CPU operations by time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # 8. Memory timeline
    print("\n" + "=" * 60)
    print("8. MEMORY TIMELINE")
    print("=" * 60)
    reset_peak_memory()
    mem_snapshots = []

    stages = [
        ("after model", lambda: None),
        ("after forward", lambda: model(x, targets=y, training_mode=use_composite)),
        ("after backward", lambda: (
            (lambda o: (
                (o["loss"] if "loss" in o else o["logits"]) / grad_accum
            ).backward()
        )(model(x, targets=y, training_mode=use_composite)) or None),
        ),
        ("after optimizer", lambda: optimizer.step() or None),
    ]

    for stage_name, stage_fn in stages:
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        reset_peak_memory()
        stage_fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        mem = get_gpu_memory()
        mem_snapshots.append((stage_name, mem))
        print(f"  {stage_name:20s}: allocated={mem['allocated']:.2f}GB  peak={mem['max_allocated']:.2f}GB")

    # 9. Summary
    print("\n" + "=" * 60)
    print("9. SUMMARY")
    print("=" * 60)
    total_step = fwd_time + bwd_time + (opt_time - fwd_time - bwd_time)
    # Actually opt_time is full step, so:
    step_time = opt_time
    tok_per_sec = (batch_size * seq_len) / step_time
    print(f"Forward pass:   {format_time_ms(fwd_time)}")
    print(f"Backward pass:  {format_time_ms(bwd_time)}")
    print(f"Optimizer step: {format_time_ms(step_time - fwd_time - bwd_time)}")
    print(f"Full step:      {format_time_ms(step_time)}")
    print(f"Tokens/sec:     {tok_per_sec:.0f}")
    print(f"GPU utilization: limited by CPU overhead (see profiler table)")
    print(f"Bottleneck: check 'Top 20 CUDA operations' above")

    # 10. Per-module parameter count
    print("\n" + "=" * 60)
    print("10. MODULE PARAMETER COUNTS")
    print("=" * 60)
    module_params = {}
    for name, param in model.named_parameters():
        top_module = name.split(".")[0]
        module_params[top_module] = module_params.get(top_module, 0) + param.numel()
    for name, count in sorted(module_params.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / params
        print(f"  {name:20s}: {count:>12,} ({pct:5.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default="configs/rtx3070_canonical.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=Path, default="data/fineweb_1M")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    profile_full_training(args)


if __name__ == "__main__":
    main()
