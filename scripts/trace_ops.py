"""Operator-level profiler: prints top GPU ops by self-time over N steps.

Isolates where fwd+bwd time goes (attention vs mlp vs moe vs gdr vs hrm vs ce).
Single process only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parent / "src"))

import train as train_script  # noqa: E402
from hagi.model import HAGI  # noqa: E402
from hagi.train.config import config_from_dict  # noqa: E402
from hagi.train.loop import _resolve_loss, autocast_ctx  # noqa: E402


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/rtx3070_canonical.yaml"))
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = config_from_dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("training", {}))
    data_cfg = cfg.get("data", {})
    device = "cuda"

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = True

    grad_accum = int(train_cfg.get("grad_accum_steps", 2))
    batch_size = int(train_cfg.get("batch_size", 8))
    seq_len = int(data_cfg.get("max_seq_len", 1024))
    chunk_size = int(getattr(model_cfg, "ce_chunk_size", 0))

    model = HAGI(model_cfg).to(device).to(torch.bfloat16)
    model.cfg.gradient_checkpointing = bool(
        train_cfg.get("gradient_checkpointing", model_cfg.gradient_checkpointing)
    )
    precision = "manual_bf16"
    train_model = model
    train_model.train()
    optimizer = train_script.build_optimizer(model, train_cfg)
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]

    composite_cfg = train_cfg.get("composite_loss")
    composite_weights = dict(composite_cfg) if isinstance(composite_cfg, dict) else None
    vocab = int(getattr(model_cfg, "vocab_size", 49152))

    def get_batch():
        x = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        return x, y

    # warmup
    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = get_batch()
            with autocast_ctx(precision, device):
                out = train_model(x, targets=y, training_mode=composite_weights is not None, weights=composite_weights)
                loss, _ = _resolve_loss(out, y, composite_weights, chunk_size)
                loss = loss / grad_accum
            loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            with record_function("forward_stage"):
                for _ in range(grad_accum):
                    x, y = get_batch()
                    with autocast_ctx(precision, device):
                        with record_function("fwd"):
                            out = train_model(x, targets=y, training_mode=composite_weights is not None, weights=composite_weights)
                        with record_function("loss"):
                            loss, _ = _resolve_loss(out, y, composite_weights, chunk_size)
                            loss = loss / grad_accum
            with record_function("backward_stage"):
                loss.backward()
            with record_function("optim_stage"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
    torch.cuda.synchronize()

    print("\n===== TOP CUDA OPS (self time, avg us) =====")
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", row_limit=25))


if __name__ == "__main__":
    main()
