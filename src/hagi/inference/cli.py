from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

try:
    import typer
except ImportError:  # pragma: no cover - dependency fallback
    typer = None

import torch

from hagi.data import TokenizerWrapper
from hagi.inference.chat import ChatSession
from hagi.model import HAGI
from hagi.train.config import config_from_dict


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _load_model(checkpoint: Path, config: Path, device: str) -> HAGI:
    cfg = _load_yaml(config)
    model = HAGI(config_from_dict(cfg.get("model", cfg)))
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run(checkpoint: Path, config: Path, device: str = "cpu") -> None:
    model = _load_model(checkpoint, config, device)
    tokenizer = TokenizerWrapper()
    session = ChatSession(model, tokenizer)
    print("HAGI chat. Type /quit to exit.")
    while True:
        try:
            text = input("user> ")
        except EOFError:
            break
        if text.strip() == "/quit":
            break
        session.add_user_message(text)
        response = session.generate_response()
        print(f"assistant> {response}")


def main() -> None:
    if typer is not None:
        def cli(
            checkpoint: Path = typer.Option(..., "--checkpoint"),
            config: Path = typer.Option(..., "--config"),
            device: str = typer.Option("cpu", "--device"),
        ) -> None:
            run(checkpoint, config, device)

        typer.run(cli)
        return

    import argparse

    parser = argparse.ArgumentParser(prog="hagi-chat")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
