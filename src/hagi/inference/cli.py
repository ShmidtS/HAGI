from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import typer
except ImportError:  # pragma: no cover - dependency fallback
    typer: Any = None

from hagi.data import TokenizerWrapper
from hagi.inference.chat import ChatSession
from hagi.model import HAGI
from hagi.train.loop import load_checkpoint


def _load_model(checkpoint: Path, config: Path | None = None, device: str = "cpu") -> HAGI:
    """Load a HAGI model from a checkpoint, preferring EMA weights.

    Sharded layout: ``<dir>/{model.pt,optimizer.pt,ema.pt,meta.pt}``.
    EMA weights (``ema.pt``) are preferred for inference — they are a
    Polyak-Ruppert average that is smoother and gives better generation
    quality than the raw main weights. Falls back to ``model.pt`` when
    ``ema.pt`` is absent (e.g. EMA disabled or early checkpoint).

    The ``config`` argument is accepted for CLI compatibility but unused —
    the model config is rebuilt from the checkpoint's ``meta.pt``.
    """
    model, _step, _ema = load_checkpoint(
        str(checkpoint),
        device=device,
        use_ema=True,
    )
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
