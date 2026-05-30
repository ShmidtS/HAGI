from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import torch

from hagi.inference.generate import generate
from hagi.model import HAGI, HAGIConfig, TransformerConfig
from hagi.train.checkpoint import load_checkpoint, save_checkpoint
from hagi.train.loop import LoopConfig, train

CHECKPOINT_PATH = Path("E:/HAGI/checkpoints/toy_chat.pt")
VOCAB_SIZE = 64
SEQ_LEN = 40

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
TEXT_TOKENS = list("What is?Answer. 0123456789\n")
TOKEN_TO_ID = {token: idx for idx, token in enumerate(SPECIAL_TOKENS + TEXT_TOKENS)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}
PAD_ID = TOKEN_TO_ID["<pad>"]
BOS_ID = TOKEN_TO_ID["<bos>"]
EOS_ID = TOKEN_TO_ID["<eos>"]
UNK_ID = TOKEN_TO_ID["<unk>"]


def encode(text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    ids = [TOKEN_TO_ID.get(char, UNK_ID) for char in text]
    if add_bos:
        ids.insert(0, BOS_ID)
    if add_eos:
        ids.append(EOS_ID)
    return ids


def decode(ids: list[int]) -> str:
    chars = []
    for token_id in ids:
        if token_id in (PAD_ID, BOS_ID):
            continue
        if token_id == EOS_ID:
            break
        token = ID_TO_TOKEN.get(int(token_id), "")
        chars.append("?" if token == "<unk>" else token)
    return "".join(chars)


def build_dataset(size: int = 100) -> list[str]:
    return [f"What is {idx}?\nAnswer is {idx}." for idx in range(size)]


def make_batcher(dataset: list[str], batch_size: int, device: str):
    encoded = [encode(example, add_bos=True, add_eos=True) for example in dataset]

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.full((batch_size, SEQ_LEN), PAD_ID, dtype=torch.long)
        ys = torch.full((batch_size, SEQ_LEN), -100, dtype=torch.long)
        for row, ids in enumerate(random.choices(encoded, k=batch_size)):
            ids = ids[: SEQ_LEN + 1]
            x = ids[:-1]
            y = ids[1:]
            xs[row, : len(x)] = torch.tensor(x, dtype=torch.long)
            ys[row, : len(y)] = torch.tensor(y, dtype=torch.long)
        return xs.to(device), ys.to(device)

    return get_batch


def build_model() -> HAGI:
    transformer = TransformerConfig(
        hidden_size=32,
        num_query_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=SEQ_LEN + 32,
    )
    cfg = HAGIConfig(
        vocab_size=VOCAB_SIZE,
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
    model = build_model()
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
    get_batch = make_batcher(build_dataset(), batch_size=16, device=device)
    final_loss = train(model, optimizer, get_batch, loop_cfg, device=device)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, step=steps, ckpt_dir=str(checkpoint_path.parent))
    step_path = checkpoint_path.parent / f"step-{steps:08d}.pt"
    shutil.move(str(step_path), checkpoint_path)
    print(f"saved toy checkpoint -> {checkpoint_path}")
    print(f"final_loss {final_loss:.4f}")


def answer(model: HAGI, question: str, device: str, max_new_tokens: int) -> str:
    prompt = question.strip()
    if not prompt.endswith("?"):
        prompt = f"What is {prompt}?" if prompt.isdigit() else f"{prompt}?"
    prompt = f"{prompt}\nAnswer"
    prompt_ids = torch.tensor([encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    output = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=None,
        top_p=None,
        eos_token_id=EOS_ID,
    )
    generated_ids = output[0, prompt_ids.shape[1] :].tolist()
    text = decode(generated_ids).strip()
    return f"Answer{text}" if text else ""


def chat(checkpoint_path: Path, device: str, max_new_tokens: int) -> None:
    model, step = load_checkpoint(str(checkpoint_path), device=device)
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
        print(f"hagi> {answer(model, question, device, max_new_tokens)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and chat with a tiny HAGI toy model.")
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train and not args.checkpoint.exists():
        train_toy(args.steps, args.checkpoint, args.device)
    elif args.checkpoint.exists():
        print(f"checkpoint exists, skipping training: {args.checkpoint}")
    elif args.chat:
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    if args.chat:
        chat(args.checkpoint, args.device, args.max_new_tokens)


if __name__ == "__main__":
    main()
