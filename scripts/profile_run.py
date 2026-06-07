"""Profile a short training run with PyTorch profiler.

Usage:
    python scripts/profile_run.py --config configs/rtx3070_canonical.yaml --max-steps 20

Saves chrome trace to .omc/ultragoal/profiler_trace.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.profiler

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib.util
_train_spec = importlib.util.spec_from_file_location("train", str(Path(__file__).resolve().parent / "train.py"))
assert _train_spec is not None
_train_mod = importlib.util.module_from_spec(_train_spec)
sys.modules["train"] = _train_mod
assert _train_spec.loader is not None
_train_spec.loader.exec_module(_train_mod)

load_yaml = _train_mod.load_yaml
detect_mode = _train_mod.detect_mode
run_full = _train_mod.run_full
run_fast = _train_mod.run_fast
run_basic = _train_mod.run_basic


def profile_full(args, cfg):
    """Run full mode with PyTorch profiler for a few steps."""
    from hagi.model import HAGI
    from hagi.train.config import config_from_dict
    from hagi.train.optim import build_optimizer
    from hagi.train.loop import save_checkpoint
    from hagi.data import get_memmap_dataloader, PrefixLMBatch
    resolve_train_path = _train_mod.resolve_train_path
    build_full_dataloader = _train_mod.build_full_dataloader
    print_model_summary = _train_mod.print_model_summary
    run_dry_profile = _train_mod.run_dry_profile
    to_device = _train_mod.to_device
    apply_prefix_mask = _train_mod.apply_prefix_mask
    unwrap_logits = _train_mod.unwrap_logits
    compute_loss = _train_mod.compute_loss
    get_grad_norm = _train_mod.get_grad_norm
    update_ema = _train_mod.update_ema
    magic_norm_clip = _train_mod.magic_norm_clip
    save_training_checkpoint = _train_mod.save_training_checkpoint
    gpu_util = _train_mod.gpu_util
    lr_at = _train_mod.lr_at
    scheduled_weight = _train_mod.scheduled_weight
    autocast_ctx = _train_mod.autocast_ctx
    maybe_compile = _train_mod.maybe_compile
    resolve_mix_paths = _train_mod.resolve_mix_paths

    model_cfg = config_from_dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("training", {}))
    data_cfg = cfg.get("data", {})
    device = args.device

    if device.startswith("cuda"):
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = HAGI(model_cfg).to(device)
    train_model = maybe_compile(model, device)
    train_model.train()

    use_prefix_lm = bool(train_cfg.get("use_prefix_lm", False))
    composite_cfg = train_cfg.get("composite_loss")
    composite_weights = dict(composite_cfg) if isinstance(composite_cfg, dict) else None
    dataset_mode = args.dataset_mode or str(data_cfg.get("dataset_mode", "memmap"))
    train_path = (
        resolve_train_path(cfg, args.train_path, args.data_dir)
        if args.train_path is not None or not resolve_mix_paths(data_cfg, args.data_dir)
        else None
    )
    eval_samples = int(train_cfg.get("eval_samples", 500))
    dataloader, eval_loader, batch_size, seq_len, pin_memory = build_full_dataloader(
        cfg, train_path, args.data_dir, use_prefix_lm, device, eval_samples=eval_samples, dataset_mode=dataset_mode,
    )
    data_iter = iter(dataloader)

    max_steps = int(args.max_steps if args.max_steps is not None else 20)
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 4))
    warmup_steps = int(train_cfg.get("warmup_steps", 500))
    learning_rate = float(train_cfg.get("learning_rate", 5.0e-4))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    precision = str(train_cfg.get("precision", "fp16"))
    log_interval = int(train_cfg.get("log_interval", 25))
    use_scaler = precision == "fp16" and device.startswith("cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    optimizer = build_optimizer(model, train_cfg)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    print_model_summary(model, model_cfg, device, use_prefix_lm, composite_weights is not None)

    # Profile config
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    profile_dir = Path(".omc/ultragoal")
    profile_dir.mkdir(parents=True, exist_ok=True)
    trace_path = profile_dir / "profiler_trace.json"
    mem_path = profile_dir / "profiler_memory.txt"

    prof = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
    )

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    start_mem = torch.cuda.memory_allocated(device) / 1024**3 if device.startswith("cuda") else 0

    prof.start()
    for step in range(max_steps):
        lr = lr_at(step, max_steps, warmup_steps, learning_rate, min_lr_ratio)
        ratio = lr / max(learning_rate, 1e-12)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * ratio

        optimizer.zero_grad(set_to_none=True)
        accum_loss_tensor = None
        for _ in range(grad_accum_steps):
            try:
                batch, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch, targets = next(data_iter)

            batch = to_device(batch, device, pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)
            targets = apply_prefix_mask(targets, batch)
            tokens = batch.tokens if isinstance(batch, PrefixLMBatch) else batch
            with autocast_ctx(precision, device):
                output = train_model(tokens, targets=targets, training_mode=composite_weights is not None)
                logits = unwrap_logits(output)
                chunk_size = getattr(model_cfg, 'ce_chunk_size', 0)
                loss, _ = compute_loss(logits, targets, output, composite_weights, chunk_size=chunk_size)
                loss = loss / grad_accum_steps
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss_tensor = loss.detach() if accum_loss_tensor is None else accum_loss_tensor + loss.detach()

        if use_scaler:
            scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        prof.step()
        if step % log_interval == 0:
            print(f"step {step} | profile active")

    prof.stop()
    prof.export_chrome_trace(str(trace_path))
    del prof
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"trace saved -> {trace_path}")

    if device.startswith("cuda"):
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        with open(mem_path, "w") as f:
            f.write(f"start_mem_gb: {start_mem:.2f}\n")
            f.write(f"peak_mem_gb: {peak_mem:.2f}\n")
        print(f"memory saved -> {mem_path}")

    print("profile complete")


def main():
    parser = argparse.ArgumentParser(prog="profile_run")
    parser.add_argument("--config", type=Path, default=Path("configs/rtx3070_canonical.yaml"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--dataset-mode", choices=["memmap", "memmap_packed", "sft"], default=None)
    parser.add_argument("--mode", choices=["auto", "basic", "fast", "full"], default="auto")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    mode = args.mode if args.mode != "auto" else detect_mode(cfg)

    if mode == "full":
        profile_full(args, cfg)
    else:
        print(f"profile mode {mode} not yet implemented, use --mode full")
        sys.exit(1)


if __name__ == "__main__":
    main()
