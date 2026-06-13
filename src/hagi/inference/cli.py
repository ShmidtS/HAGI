from __future__ import annotations

from pathlib import Path
try:
    import typer
except ImportError:  # pragma: no cover - dependency fallback
    typer = None

import torch

from hagi.data import TokenizerWrapper
from hagi.inference.chat import ChatSession
from hagi.model import HAGI
from hagi.train.config import config_from_dict
from hagi.utils import _load_yaml


def _load_state_dict(checkpoint: Path, device: str) -> tuple[dict, dict]:
    """Load (model_state_dict, aux_state) from a flat file or sharded dir.

    Sharded layout: ``<dir>/{model.pt,meta.pt,...}``. The model weights always
    come from ``model.pt`` (never ``ema.pt`` — EMA in this checkpoint is stale).
    """
    model_pt = checkpoint / "model.pt" if checkpoint.is_dir() else checkpoint
    state = torch.load(model_pt, map_location=device, weights_only=True)
    # Flat checkpoints wrap weights under {"model": ...}; sharded dirs store
    # a raw state_dict at the top level.
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        aux = state
        state_dict = state["model"]
    else:
        aux = state if isinstance(state, dict) else {}
        state_dict = state
    return state_dict, aux


def _load_model(checkpoint: Path, config: Path, device: str) -> HAGI:
    cfg = _load_yaml(config)
    model = HAGI(config_from_dict(cfg.get("model", cfg)))
    state_dict, state = _load_state_dict(checkpoint, device)
    model.load_state_dict(state_dict)
    # Load MSA and NARS states if present
    if hasattr(model, "msa_registry") and model.msa_registry is not None and "msa_registry" in state:
        model.msa_registry.load_state_dict(state["msa_registry"])
    if hasattr(model, "nars_hrm") and model.nars_hrm is not None and "nars_hrm" in state:
        model.nars_hrm.load_state_dict(state["nars_hrm"])
    if hasattr(model, "nars_hdim") and model.nars_hdim is not None and "nars_hdim" in state:
        model.nars_hdim.load_state_dict(state["nars_hdim"])
    if hasattr(model, "nars_msa") and model.nars_msa is not None and "nars_msa" in state:
        model.nars_msa.load_state_dict(state["nars_msa"])
    model.to(device)
    model.eval()
    return model


def run(checkpoint: Path, config: Path, device: str = "cpu") -> None:
    model = _load_model(checkpoint, config, device)
    tokenizer = TokenizerWrapper.smollm2()
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
