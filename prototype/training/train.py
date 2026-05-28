"""Training entry point.

Usage:
    python -m prototype.training.train --config configs/gdr.yaml

This is the Stage 0/1/2 training driver. The data pipeline (streaming tokenized
shards) is a TODO — wire in FineWeb-Edu + code + math once the tokenizer and
sharding are set up. The loop below is the skeleton: build model, optimizer,
scheduler, step, log, checkpoint, eval-hook.
"""

from __future__ import annotations

import argparse

import torch

from prototype.model.hagi import HAGI
from prototype.training.config import load_config


def build_optimizer(model: torch.nn.Module, cfg: dict):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "iter_embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.get("weight_decay", 0.1)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.get("learning_rate", 3e-4),
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def lambda_iso(step: int, cfg: dict) -> float:
    target = cfg.get("lambda_iso_target", 0.0)
    warmup = int(cfg.get("iso_warmup_ratio", 0.2) * cfg.get("max_steps", 1))
    if warmup <= 0:
        return target
    return target * min(1.0, step / warmup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["training"]
    torch.manual_seed(tcfg.get("seed", 42))

    device = args.device
    model = HAGI(cfg["model"]).to(device)
    print(f"[{cfg['name']}] parameters: {model.num_parameters() / 1e6:.1f}M")

    _optimizer = build_optimizer(model, tcfg)  # noqa: F841  (used once data pipeline lands)
    # scheduler = ...  # cosine with warmup (TODO)

    # TODO: data pipeline.
    #   from prototype.data.loader import build_dataloader
    #   loader = build_dataloader(cfg["data"], tcfg["batch_size"])
    raise SystemExit(
        "Data pipeline not yet implemented. See prototype/data/ TODOs.\n"
        "Model builds and is ready: " + f"{model.num_parameters() / 1e6:.1f}M params."
    )


if __name__ == "__main__":
    main()
