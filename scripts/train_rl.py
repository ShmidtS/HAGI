"""RL training script — MGPO on HAGI (VibeThinker-inspired).

Usage:
    python scripts/train_rl.py --config configs/rl_rtx3070.yaml \
        --sft-checkpoint checkpoints/rtx3070/step-146000.pt

Loads an SFT checkpoint, builds a verifiable-reward RL dataset, and runs
the MGPO training loop. Designed for RTX 3070 8GB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from hagi.data.tokenizer import TokenizerWrapper
from hagi.model import HAGI
from hagi.train.config import config_from_dict
from hagi.train.rl_loop import RLConfig, train_rl
from hagi.utils import _load_yaml as load_yaml

ROOT = Path(__file__).resolve().parents[1]


def load_rl_dataset(
    dataset_path: Path,
    tokenizer: TokenizerWrapper,
    max_prompt_len: int,
) -> list[dict[str, Any]]:
    """Load RL prompts from a JSONL file.

    Each line: {"prompt": str, "reference": str, "domain": str (optional)}
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"RL dataset not found: {dataset_path}. "
            "Create a JSONL file with lines: "
            '{"prompt": "...", "reference": "..."}'
        )

    dataset: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            prompt = entry.get("prompt", "")
            reference = entry.get("reference", entry.get("answer", ""))
            if not prompt or not reference:
                continue
            prompt_ids = tokenizer.encode(prompt)
            if isinstance(prompt_ids, list):
                prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
            if prompt_ids.size(0) > max_prompt_len:
                prompt_ids = prompt_ids[:max_prompt_len]
            dataset.append(
                {
                    "prompt_ids": prompt_ids,
                    "reference": str(reference),
                    "domain": entry.get("domain", "math"),
                }
            )
    print(f"RL dataset: {len(dataset)} prompts from {dataset_path}")
    return dataset


def build_rl_config(rl_cfg: dict[str, Any]) -> RLConfig:
    """Build RLConfig from config dict."""
    return RLConfig(
        max_steps=int(rl_cfg.get("max_steps", 5000)),
        group_size=int(rl_cfg.get("group_size", 4)),
        num_prompts_per_step=int(rl_cfg.get("num_prompts_per_step", 4)),
        max_new_tokens=int(rl_cfg.get("max_new_tokens", 256)),
        min_prompt_len=int(rl_cfg.get("min_prompt_len", 16)),
        temperature=float(rl_cfg.get("temperature", 1.0)),
        top_k=rl_cfg.get("top_k", 50),
        top_p=rl_cfg.get("top_p", 0.9),
        learning_rate=float(rl_cfg.get("learning_rate", 1e-5)),
        min_lr_ratio=float(rl_cfg.get("min_lr_ratio", 0.1)),
        warmup_steps=int(rl_cfg.get("warmup_steps", 100)),
        clip_eps=float(rl_cfg.get("clip_eps", 0.2)),
        mgpo_gamma=float(rl_cfg.get("mgpo_gamma", 4.0)),
        mgpo_p0=float(rl_cfg.get("mgpo_p0", 0.5)),
        long2short_lambda=float(rl_cfg.get("long2short_lambda", 0.0)),
        min_accuracy=float(rl_cfg.get("min_accuracy", 0.0)),
        max_accuracy=float(rl_cfg.get("max_accuracy", 1.0)),
        grad_accum_steps=int(rl_cfg.get("grad_accum_steps", 1)),
        grad_clip=float(rl_cfg.get("grad_clip", 1.0)),
        ckpt_interval=int(rl_cfg.get("ckpt_interval", 500)),
        log_interval=int(rl_cfg.get("log_interval", 10)),
        eval_interval=int(rl_cfg.get("eval_interval", 500)),
        ckpt_dir=rl_cfg.get("ckpt_dir", "checkpoints/rl"),
        seed=int(rl_cfg.get("seed", 42)),
        precision=rl_cfg.get("precision", "manual_bf16"),
        gradient_checkpointing=bool(
            rl_cfg.get("gradient_checkpointing", True)
        ),
        entropy_coeff=float(rl_cfg.get("entropy_coeff", 0.01)),
        update_epochs=int(rl_cfg.get("update_epochs", 1)),
        reference_model_free=bool(
            rl_cfg.get("reference_model_free", True)
        ),
    )


def build_optimizer(
    model: HAGI, opt_cfg: dict[str, Any]
) -> torch.optim.Optimizer:
    """Build AdamW optimizer for RL training."""
    lr = float(opt_cfg.get("learning_rate", 1e-5))
    wd = float(opt_cfg.get("weight_decay", 0.01))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.95]))
    eps = float(opt_cfg.get("eps", 1e-8))
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=betas,
        eps=eps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="train_rl")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "rl_rtx3070.yaml"
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=Path,
        default=None,
        help="Path to SFT checkpoint to start RL from",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to RL checkpoint to resume from",
    )
    parser.add_argument(
        "--rl-dataset",
        type=Path,
        default=None,
        help="Override RL dataset path from config",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    sft_ckpt = args.sft_checkpoint or Path(
        cfg.get("sft_checkpoint", "checkpoints/rtx3070/step-146000.pt")
    )
    if not sft_ckpt.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found: {sft_ckpt}. "
            "Train SFT first or specify --sft-checkpoint."
        )

    model_cfg = config_from_dict(cfg.get("model", {}))
    model = HAGI(model_cfg).to(args.device)

    if args.resume is not None and args.resume.exists():
        state = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.load_state_dict(state["model"], strict=False)
        start_step = int(state.get("step", 0))
        print(f"Resumed RL from {args.resume} at step {start_step}")
    else:
        state = torch.load(sft_ckpt, map_location=args.device, weights_only=True)
        model_state = state.get("model", state)
        model.load_state_dict(model_state, strict=False)
        start_step = 0
        print(f"Loaded SFT checkpoint: {sft_ckpt}")

    if cfg.get("rl", {}).get("precision") == "manual_bf16":
        model = model.to(torch.bfloat16)
        print("Model cast to bf16 (manual_bf16 mode)")

    data_cfg = cfg.get("data", {})
    tokenizer_name = str(
        data_cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M")
    )
    tokenizer = TokenizerWrapper.smollm2(tokenizer_name, use_fast=True)

    dataset_path = args.rl_dataset or Path(
        data_cfg.get("rl_dataset_path", "data/rl_prompts.jsonl")
    )
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    max_prompt_len = int(data_cfg.get("max_prompt_len", 512))
    rl_dataset = load_rl_dataset(dataset_path, tokenizer, max_prompt_len)

    if len(rl_dataset) == 0:
        raise ValueError(
            "RL dataset is empty. Create a JSONL file with verifiable prompts."
        )

    rl_cfg = build_rl_config(cfg.get("rl", {}))
    opt_cfg = cfg.get("optimizer", {})
    optimizer = build_optimizer(model, opt_cfg)

    from hagi.train.rewards import math_reward_fn

    eos_token_id = tokenizer.tokenizer.eos_token_id

    final_reward = train_rl(
        model=model,
        optimizer=optimizer,
        rl_dataset=rl_dataset,
        cfg=rl_cfg,
        device=args.device,
        start_step=start_step,
        reward_fn=math_reward_fn,
        eos_token_id=eos_token_id,
    )

    final_ckpt = Path(rl_cfg.ckpt_dir) / "final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": rl_cfg.max_steps,
            "rl_config": rl_cfg.__dict__,
            "final_reward": final_reward,
        },
        final_ckpt,
    )
    print(f"RL training complete. Final reward: {final_reward:.4f}")
    print(f"Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
