from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import torch

from hagi.data import TokenizerWrapper
from hagi.inference.chat import ChatSession
from hagi.inference.generate import generate
from hagi.model import HAGI, HAGIConfig, TransformerConfig
from hagi.train.checkpoint import load_checkpoint, save_checkpoint
from hagi.train.loop import LoopConfig, train


DEFAULT_CONFIG_PATH = Path("E:/HAGI/configs/rtx3070_canonical.yaml")
DEFAULT_CHECKPOINT_DIR = Path("E:/HAGI/checkpoints/rtx3070")
TOY_CHECKPOINT_PATH = Path("E:/HAGI/checkpoints/toy_chat.pt")

TOY_VOCAB_SIZE = 64
TOY_SEQ_LEN = 40
TOY_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
TOY_TEXT_TOKENS = list("What is?Answer. 0123456789\n")
TOY_TOKEN_TO_ID = {token: idx for idx, token in enumerate(TOY_SPECIAL_TOKENS + TOY_TEXT_TOKENS)}
TOY_ID_TO_TOKEN = {idx: token for token, idx in TOY_TOKEN_TO_ID.items()}
TOY_PAD_ID = TOY_TOKEN_TO_ID["<pad>"]
TOY_BOS_ID = TOY_TOKEN_TO_ID["<bos>"]
TOY_EOS_ID = TOY_TOKEN_TO_ID["<eos>"]
TOY_UNK_ID = TOY_TOKEN_TO_ID["<unk>"]


def toy_encode(text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    ids = [TOY_TOKEN_TO_ID.get(char, TOY_UNK_ID) for char in text]
    if add_bos:
        ids.insert(0, TOY_BOS_ID)
    if add_eos:
        ids.append(TOY_EOS_ID)
    return ids


def toy_decode(ids: list[int]) -> str:
    chars = []
    for token_id in ids:
        if token_id in (TOY_PAD_ID, TOY_BOS_ID):
            continue
        if token_id == TOY_EOS_ID:
            break
        token = TOY_ID_TO_TOKEN.get(int(token_id), "")
        chars.append("?" if token == "<unk>" else token)
    return "".join(chars)


def toy_build_dataset(size: int = 100) -> list[str]:
    return [f"What is {idx}?\nAnswer is {idx}." for idx in range(size)]


def toy_make_batcher(dataset: list[str], batch_size: int, device: str):
    encoded = [toy_encode(example, add_bos=True, add_eos=True) for example in dataset]

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.full((batch_size, TOY_SEQ_LEN), TOY_PAD_ID, dtype=torch.long)
        ys = torch.full((batch_size, TOY_SEQ_LEN), -100, dtype=torch.long)
        for row, ids in enumerate(random.choices(encoded, k=batch_size)):
            ids = ids[: TOY_SEQ_LEN + 1]
            x = ids[:-1]
            y = ids[1:]
            xs[row, : len(x)] = torch.tensor(x, dtype=torch.long)
            ys[row, : len(y)] = torch.tensor(y, dtype=torch.long)
        return xs.to(device), ys.to(device)

    return get_batch


def toy_build_model() -> HAGI:
    transformer = TransformerConfig(
        hidden_size=32,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=TOY_SEQ_LEN + 32,
    )
    cfg = HAGIConfig(
        vocab_size=TOY_VOCAB_SIZE,
        hidden_size=32,
        perception_layers=1,
        reasoning_layers=1,
        expression_layers=1,
        loop_count=1,
        use_loop=False,
        use_gdr=False,
        transformer=transformer,
    )
    return HAGI(cfg)


def train_toy(steps: int, checkpoint_path: Path, device: str) -> None:
    torch.manual_seed(0)
    random.seed(0)
    model = toy_build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loop_cfg = LoopConfig(
        max_steps=steps,
        warmup_steps=10,
        learning_rate=3e-3,
        min_lr_ratio=0.2,
        grad_accum_steps=1,
        grad_clip=1.0,
        precision="fp32",
        eval_interval=0,
        ckpt_interval=0,
        log_interval=max(1, steps // 10),
        ckpt_dir=str(checkpoint_path.parent),
    )
    get_batch = toy_make_batcher(toy_build_dataset(), batch_size=16, device=device)
    final_loss = train(model, optimizer, get_batch, loop_cfg, device=device)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, step=steps, ckpt_dir=str(checkpoint_path.parent))
    step_path = checkpoint_path.parent / f"step-{steps:08d}.pt"
    shutil.move(str(step_path), checkpoint_path)
    print(f"saved toy checkpoint -> {checkpoint_path}")
    print(f"final_loss {final_loss:.4f}")


def toy_answer(model: HAGI, question: str, device: str, max_new_tokens: int) -> str:
    prompt = question.strip()
    if not prompt.endswith("?"):
        prompt = f"What is {prompt}?" if prompt.isdigit() else f"{prompt}?"
    prompt = f"{prompt}\nAnswer"
    prompt_ids = torch.tensor([toy_encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    output = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=None,
        top_p=None,
        eos_token_id=TOY_EOS_ID,
    )
    generated_ids = output[0, prompt_ids.shape[1] :].tolist()
    text = toy_decode(generated_ids).strip()
    return f"Answer{text}" if text else ""


def toy_chat(checkpoint_path: Path, device: str, max_new_tokens: int) -> None:
    model, step, _ = load_checkpoint(str(checkpoint_path), device=device)
    model.eval()
    print(f"loaded checkpoint from step {step}: {checkpoint_path}")
    print("Type /quit to exit.")
    while True:
        try:
            question = input("you> ").strip()
        except EOFError:
            break
        if question == "/quit":
            break
        if not question:
            continue
        print(f"hagi> {toy_answer(model, question, device, max_new_tokens)}")


def find_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    # Sharded checkpoint directory (model.pt, optimizer.pt, ema.pt, meta.pt)
    if (path / "model.pt").exists():
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


def load_production_model(checkpoint: Path, compile_model: bool, use_msa: bool = True, use_nars: bool = True) -> tuple[torch.nn.Module, int]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, step, _ = load_checkpoint(str(checkpoint), device=device)
    model.eval()
    # Enable MSA and NARS for inference if configured
    if hasattr(model.cfg, "use_msa"):
        model.cfg.use_msa = use_msa
    if hasattr(model.cfg, "use_nars"):
        model.cfg.use_nars = use_nars
    if compile_model and device == "cuda" and hasattr(torch, "compile") and sys.platform != "win32":
        model = torch.compile(model)
        assert isinstance(model, torch.nn.Module)
    return model, step


def production_repl(args: argparse.Namespace) -> None:
    checkpoint = find_checkpoint(args.checkpoint)
    model, step = load_production_model(checkpoint, args.compile)
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
    session.rollouts = args.rollouts
    session.noise_sigma = args.noise_sigma
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
    parser = argparse.ArgumentParser(description="HAGI chat REPL (production + toy).")
    parser.add_argument("--config", type=Path, default=None, help="config path (production mode)")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", choices=["auto", "production", "toy"], default="auto")
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system", default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    return parser.parse_args()


def detect_chat_mode(args: argparse.Namespace) -> str:
    if args.mode != "auto":
        return args.mode
    if args.config is not None:
        return "production"
    if args.checkpoint is not None and str(args.checkpoint).endswith("toy_chat.pt"):
        return "toy"
    if TOY_CHECKPOINT_PATH.exists():
        return "toy"
    return "production"


def main() -> None:
    args = parse_args()
    mode = detect_chat_mode(args)

    if mode == "toy":
        checkpoint = args.checkpoint or TOY_CHECKPOINT_PATH
        if args.train and not checkpoint.exists():
            train_toy(args.steps, checkpoint, args.device)
        elif checkpoint.exists():
            print(f"checkpoint exists, skipping training: {checkpoint}")
        elif args.chat:
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

        if args.chat:
            toy_chat(checkpoint, args.device, args.max_new_tokens)
        return

    if mode == "production":
        checkpoint = args.checkpoint or DEFAULT_CHECKPOINT_DIR
        args.checkpoint = checkpoint
        production_repl(args)
        return

    raise ValueError(f"unknown chat mode: {mode}")


if __name__ == "__main__":
    main()
