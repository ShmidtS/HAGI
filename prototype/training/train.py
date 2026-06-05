"""Training entry point.

Usage:
    python -m prototype.training.train --config configs/gdr.yaml --data data/fineweb-edu

Wires config + data + model + optimizer into the core loop. Tokenize a corpus
first with `python -m prototype.data.tokenize ...` (see prototype/data/tokenize.py),
then point --data at the shard directory. For a fast correctness check without
real data, see tests/test_overfit.py.
"""

from __future__ import annotations

import argparse
import math

import torch

from prototype.data.dataset import make_batch_fn
from prototype.model.hagi import HAGI
from prototype.training.config import load_config
from prototype.training.loop import LoopConfig, latest_checkpoint, resume_into, train
from prototype.training.optim import build_optimizer


def _resolve_max_steps(tcfg: dict, data_cfg: dict, batch_size: int, block_size: int) -> int:
    """Single source of truth for run length.

    If `data.train_tokens` is set, derive max_steps from it so the token budget
    and step count can never drift (they previously disagreed: 5e9 annotated vs
    50000 steps = ~13.1B). Otherwise fall back to an explicit `training.max_steps`.
    """
    grad_accum = tcfg.get("grad_accum_steps", 1)
    tokens_per_step = batch_size * grad_accum * block_size
    train_tokens = data_cfg.get("train_tokens")
    if train_tokens:
        steps = math.ceil(train_tokens / tokens_per_step)
        print(f"max_steps={steps} derived from train_tokens={train_tokens:,} "
              f"({tokens_per_step:,} tokens/step)")
        return steps
    return tcfg.get("max_steps", 50000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="directory of tokenized .bin shards")
    ap.add_argument("--val-data", default=None, help="optional validation shard directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None,
                    help="checkpoint path, or 'auto' to resume the latest in <ckpt-dir>/<name>/")
    ap.add_argument("--ckpt-dir", default=None,
                    help="root dir for checkpoints (default: checkpoints/). On cloud, point at "
                         "persistent storage, e.g. /content/drive/MyDrive/hagi or /kaggle/working")
    ap.add_argument("--steps", type=int, default=None,
                    help="run at most this many steps this session, then checkpoint and exit "
                         "(checkpoint-gated local training; re-run with --resume auto to continue)")
    args = ap.parse_args()

    # TF32 matmul on Ampere+ — free ~1.3-2x on fp32 paths, harmless on older GPUs.
    torch.set_float32_matmul_precision("high")

    cfg = load_config(args.config)
    tcfg = cfg["training"]
    torch.manual_seed(tcfg.get("seed", 42))

    model = HAGI(cfg["model"]).to(args.device)
    print(f"[{cfg['name']}] parameters: {model.num_parameters() / 1e6:.1f}M")

    optimizer = build_optimizer(model, tcfg)

    # Resume must restore into the uncompiled model+optimizer before compiling.
    ckpt_dir = f"{args.ckpt_dir or 'checkpoints'}/{cfg['name']}"
    start_step = 0
    if args.resume:
        path = latest_checkpoint(ckpt_dir) if args.resume == "auto" else args.resume
        if path is not None:
            start_step = resume_into(model, optimizer, str(path), device=args.device)
            print(f"resumed from {path} at step {start_step}")
        else:
            print(f"--resume {args.resume}: no checkpoint found, starting fresh")

    # torch.compile emits fused Triton kernels (the real "use Triton"); CUDA only.
    if tcfg.get("compile", False) and args.device.startswith("cuda"):
        model = torch.compile(model)
        print("torch.compile enabled")

    block_size = cfg["data"].get("max_seq_len", 4096)
    batch_size = tcfg.get("batch_size", 16)
    get_batch = make_batch_fn(args.data, batch_size, block_size, device=args.device,
                              seed=tcfg.get("seed", 42))
    eval_get_batch = (
        make_batch_fn(args.val_data, batch_size, block_size, device=args.device, seed=123)
        if args.val_data else None
    )

    loop_cfg = LoopConfig(
        max_steps=_resolve_max_steps(tcfg, cfg["data"], batch_size, block_size),
        warmup_steps=tcfg.get("warmup_steps", 2000),
        learning_rate=tcfg.get("learning_rate", 3e-4),
        grad_accum_steps=tcfg.get("grad_accum_steps", 1),
        grad_clip=tcfg.get("grad_clip", 1.0),
        precision=tcfg.get("precision", "bf16"),
        eval_interval=cfg["eval"].get("every_n_steps", 2000),
        ckpt_interval=tcfg.get("ckpt_interval", 5000),
        ckpt_dir=ckpt_dir,
    )

    if args.steps is not None:
        print(f"session: steps {start_step}..{start_step + args.steps} (then checkpoint + exit)")

    train(model, optimizer, get_batch, loop_cfg, device=args.device,
          eval_get_batch=eval_get_batch, start_step=start_step, session_steps=args.steps)


if __name__ == "__main__":
    main()
