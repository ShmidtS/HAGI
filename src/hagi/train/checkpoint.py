"""Checkpoint helpers for HAGI training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hagi.train.loop import load_checkpoint, save_checkpoint

__all__ = ["save_checkpoint", "load_checkpoint", "save_sharded_checkpoint", "load_sharded_checkpoint"]


def save_sharded_checkpoint(
    model,
    optimizer,
    ema,
    path: Path,
    step: int | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint split into model.pt, optimizer.pt, ema.pt, meta.pt."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    if ema is not None:
        ema_state = {name: value.detach().cpu() for name, value in ema.state_dict().items()}
        torch.save(ema_state, path / "ema.pt")
    meta: dict[str, Any] = {}
    if step is not None:
        meta["step"] = step
    if config is not None:
        meta["config"] = config
    torch.save(meta, path / "meta.pt")


def load_sharded_checkpoint(
    model,
    optimizer,
    ema,
    path: Path,
) -> int:
    """Load a sharded checkpoint from model.pt, optimizer.pt, ema.pt, meta.pt."""
    model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", weights_only=True))
    if ema is not None and (path / "ema.pt").exists():
        ema.load_state_dict(torch.load(path / "ema.pt", weights_only=True))
    meta = torch.load(path / "meta.pt", weights_only=True) if (path / "meta.pt").exists() else {}
    return int(meta.get("step", 0))
