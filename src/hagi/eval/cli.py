"""Command-line entry point for HAGI evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    import typer
except ImportError:  # pragma: no cover - dependency fallback
    typer = None

from hagi.model import HAGI
from hagi.train.config import config_from_dict
from hagi.train.loop import load_checkpoint


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


@torch.no_grad()
def _golden_eval(model: HAGI) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    seq_len = min(model.cfg.transformer.max_seq_len, 16)
    x = torch.arange(seq_len, device=device).unsqueeze(0) % model.cfg.vocab_size
    y = torch.roll(x, shifts=-1, dims=1)
    logits, loss = model(x, targets=y)
    return {
        "golden": True,
        "loss": float(loss.item()),
        "logits_shape": list(logits.shape),
        "num_parameters": model.num_parameters(),
    }


def _load_model(checkpoint: Path, config: Path | None) -> tuple[HAGI, int]:
    if config is None:
        return load_checkpoint(str(checkpoint), device="cpu")

    cfg = _load_yaml(config)
    model = HAGI(config_from_dict(cfg.get("model", {})))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    step = int(state.get("step", 0)) if isinstance(state, dict) else 0
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    return model, step


def run(
    checkpoint: Path,
    config: Path | None = None,
    output: Path = Path("eval_report.json"),
    golden: bool = False,
) -> None:
    model, step = _load_model(checkpoint, config)
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "step": step,
    }
    if golden:
        report.update(_golden_eval(model))
    else:
        report.update({"golden": False, "num_parameters": model.num_parameters()})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"eval_report -> {output}")


def main() -> None:
    if typer is not None:
        def cli(
            checkpoint: Path = typer.Option(..., "--checkpoint"),
            config: Path | None = typer.Option(None, "--config"),
            output: Path = typer.Option(Path("eval_report.json"), "--output"),
            golden: bool = typer.Option(False, "--golden"),
        ) -> None:
            run(checkpoint, config, output, golden)

        typer.run(cli)
        return

    import argparse

    parser = argparse.ArgumentParser(prog="hagi-eval")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("eval_report.json"))
    parser.add_argument("--golden", action="store_true")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
