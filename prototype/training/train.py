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

import torch

from prototype.data.dataset import make_batch_fn
from prototype.model.hagi import HAGI
from prototype.training.config import load_config
from prototype.training.loop import LoopConfig, train
from prototype.training.optim import build_optimizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="directory of tokenized .bin shards")
    ap.add_argument("--val-data", default=None, help="optional validation shard directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["training"]
    torch.manual_seed(tcfg.get("seed", 42))

    model = HAGI(cfg["model"]).to(args.device)
    print(f"[{cfg['name']}] parameters: {model.num_parameters() / 1e6:.1f}M")

    optimizer = build_optimizer(model, tcfg)

    block_size = cfg["data"].get("max_seq_len", 4096)
    batch_size = tcfg.get("batch_size", 16)
    get_batch = make_batch_fn(args.data, batch_size, block_size, device=args.device,
                              seed=tcfg.get("seed", 42))
    eval_get_batch = (
        make_batch_fn(args.val_data, batch_size, block_size, device=args.device, seed=123)
        if args.val_data else None
    )

    loop_cfg = LoopConfig(
        max_steps=tcfg.get("max_steps", 50000),
        warmup_steps=tcfg.get("warmup_steps", 2000),
        learning_rate=tcfg.get("learning_rate", 3e-4),
        grad_accum_steps=tcfg.get("grad_accum_steps", 1),
        grad_clip=tcfg.get("grad_clip", 1.0),
        precision=tcfg.get("precision", "bf16"),
        eval_interval=cfg["eval"].get("every_n_steps", 2000),
        ckpt_dir=f"checkpoints/{cfg['name']}",
    )

    train(model, optimizer, get_batch, loop_cfg, device=args.device,
          eval_get_batch=eval_get_batch)


if __name__ == "__main__":
    main()
