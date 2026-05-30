from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hagi.data import TokenizerWrapper
from hagi.inference.chat import ChatSession
from hagi.train.loop import load_checkpoint

DEFAULT_CHECKPOINT_DIR = Path("E:/HAGI/checkpoints/rtx3070")


def find_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    checkpoints = sorted(path.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"no .pt checkpoints found in {path}")
    return checkpoints[0]


def vram_usage() -> str:
    if not torch.cuda.is_available():
        return "VRAM n/a"
    used = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"VRAM used={used:.2f}GB reserved={reserved:.2f}GB total={total:.2f}GB"


def load_model(checkpoint: Path, compile_model: bool) -> tuple[torch.nn.Module, int]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, step = load_checkpoint(str(checkpoint), device=device)
    if device == "cuda":
        model = model.half()
    model.eval()
    if compile_model and device == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model, step


def repl(args: argparse.Namespace) -> None:
    checkpoint = find_checkpoint(args.checkpoint)
    model, step = load_model(checkpoint, args.compile)
    tokenizer = TokenizerWrapper.smollm2()
    session = ChatSession(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        system_prompt=args.system,
        max_context_length=args.max_context_length,
        compile_model=False,
    )
    print(f"loaded checkpoint step={step}: {checkpoint}")
    print(vram_usage())
    print("Commands: /system TEXT, /clear, /quit")
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/clear":
            session.clear()
            print("history cleared")
            print(vram_usage())
            continue
        if text.startswith("/system"):
            prompt = text[len("/system") :].strip()
            session.set_system_prompt(prompt)
            print("system prompt updated" if prompt else "system prompt cleared")
            continue

        session.add_user_message(text)
        print("hagi> ", end="", flush=True)
        for piece in session.stream_response():
            print(piece, end="", flush=True)
        print()
        print(vram_usage())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTX 3070 8GB VRAM chat REPL for HAGI.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system", default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    repl(parse_args())


if __name__ == "__main__":
    main()
