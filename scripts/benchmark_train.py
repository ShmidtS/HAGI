"""Quick training benchmark for profiling optimizations.

Runs a short training burst and reports tokens/sec and memory.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from hagi.model import HAGI
from hagi.train.config import config_from_dict
from hagi.train.optim import build_optimizer


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def benchmark(config_path: Path, device: str, steps: int, batch_size: int | None, seq_len: int | None):
    cfg = load_yaml(config_path)
    model_cfg = config_from_dict(cfg.get("model", {}))
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    if device.startswith("cuda"):
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = HAGI(model_cfg).to(device)
    model.train()
    if hasattr(model.cfg, "gradient_checkpointing"):
        model.cfg.gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", True))

    bs = batch_size or int(train_cfg.get("batch_size", 8))
    sl = seq_len or int(data_cfg.get("max_seq_len", 1024))
    vocab_size = int(model_cfg.vocab_size)

    optimizer = build_optimizer(model, train_cfg)

    precision = str(train_cfg.get("precision", "bf16"))
    if precision == "manual_bf16" and device.startswith("cuda"):
        model = model.to(torch.bfloat16)
    if precision == "manual_fp16" and device.startswith("cuda"):
        model = model.half()

    grad_accum = int(train_cfg.get("grad_accum_steps", 2))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    # Synthetic data to avoid disk I/O overhead in benchmark
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    def get_batch():
        x = torch.randint(vocab_size, (bs, sl), generator=generator, device=device)
        y = torch.randint(vocab_size, (bs, sl), generator=generator, device=device)
        return x, y

    # Warmup
    for _ in range(2):
        x, y = get_batch()
        optimizer.zero_grad(set_to_none=True)
        output = model(x, targets=y, training_mode=True)
        loss = output["loss"] if isinstance(output, dict) else output[1]
        loss = loss / grad_accum
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = get_batch()
            output = model(x, targets=y, training_mode=True)
            loss = output["loss"] if isinstance(output, dict) else output[1]
            loss = loss / grad_accum
            loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    total_tokens = steps * grad_accum * bs * sl
    tokens_per_sec = total_tokens / elapsed

    result = {
        "config": config_path.name,
        "steps": steps,
        "batch_size": bs,
        "seq_len": sl,
        "grad_accum": grad_accum,
        "tokens_per_sec": tokens_per_sec,
        "elapsed_sec": elapsed,
        "total_tokens": total_tokens,
    }
    if device.startswith("cuda"):
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
        result["gpu_util"] = torch.cuda.utilization(torch.device(device).index or 0)

    print(f"BENCHMARK_RESULT {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rtx3070_canonical.yaml"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    args = parser.parse_args()
    benchmark(args.config, args.device, args.steps, args.batch_size, args.seq_len)
