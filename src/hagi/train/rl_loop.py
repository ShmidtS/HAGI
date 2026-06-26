"""MGPO (MaxEnt-Guided Policy Optimization) RL training loop for HAGI.

Adapted from VibeThinker-3B's MGPO for single-GPU 8GB constraint:
- Group size G=4 (4 rollouts per prompt)
- On-policy: generate → reward → update (no replay buffer)
- Gradient checkpointing during update phase
- Sequential rollout (one prompt at a time) to bound VRAM

Key differences from standard GRPO:
- MGPO prompt weight: w(q) = exp(-gamma * |p(q) - p0|) focuses updates on
  prompts near the model's capability boundary (p≈0.5)
- Optional Long2Short reward shift: zero-sum brevity redistribution
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
import torch.nn.functional as F

from hagi.train.rewards import (
    filter_by_difficulty,
    group_relative_advantage,
    long2short_reward_shift,
    mgpo_prompt_weight,
)

if TYPE_CHECKING:
    from hagi.model import HAGI


@dataclass
class RLConfig:
    """Configuration for the MGPO RL training loop."""

    max_steps: int = 10000
    group_size: int = 4
    num_prompts_per_step: int = 4
    max_new_tokens: int = 256
    min_prompt_len: int = 16
    temperature: float = 1.0
    top_k: int | None = 50
    top_p: float | None = 0.9
    learning_rate: float = 1e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    clip_eps: float = 0.2
    mgpo_gamma: float = 4.0
    mgpo_p0: float = 0.5
    long2short_lambda: float = 0.0
    min_accuracy: float = 0.0
    max_accuracy: float = 1.0
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    ckpt_interval: int = 1000
    log_interval: int = 10
    eval_interval: int = 500
    ckpt_dir: str = "checkpoints/rl"
    seed: int = 42
    precision: str = "manual_bf16"
    gradient_checkpointing: bool = True
    entropy_coeff: float = 0.01
    kl_coeff: float = 0.0
    update_epochs: int = 1
    reference_model_free: bool = True


def _lr_at(
    step: int, max_steps: int, warmup: int, lr: float, min_ratio: float
) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * min_ratio + coeff * lr * (1.0 - min_ratio)


@torch.no_grad()
def _generate_rollout(
    model: HAGI,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, int]:
    """Generate a single rollout response.

    Returns (full_ids, num_new_tokens) where full_ids = [prompt + response].
    """
    from hagi.inference.generate import generate

    was_training = model.training
    model.eval()
    try:
        full = generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            use_cache=True,
        )
    finally:
        if was_training:
            model.train()
    num_new = full.size(1) - prompt_ids.size(1)
    return full, max(1, num_new)


def _compute_log_probs(
    model: HAGI,
    input_ids: torch.Tensor,
    response_start: int,
    precision: str,
) -> torch.Tensor:
    """Compute log probabilities of response tokens under current policy.

    Args:
        input_ids: [1, seq_len] full sequence (prompt + response).
        response_start: index where response tokens begin.
        precision: precision mode for autocast.

    Returns:
        [response_len] log-prob tensor.
    """
    if precision == "manual_bf16" and input_ids.device.type == "cuda":
        output = model(input_ids, training_mode=False)
    else:
        with torch.autocast(
            device_type=input_ids.device.type,
            dtype=torch.bfloat16,
        ):
            output = model(input_ids, training_mode=False)

    logits = output["logits"] if isinstance(output, dict) else output[0]
    if logits is None:
        raise ValueError(
            "logits is None — disable use_fused_ce for RL training"
        )

    response_logits = logits[0, response_start - 1 : -1, :]
    response_tokens = input_ids[0, response_start:]
    log_probs = F.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)
    return token_log_probs


def train_rl(
    model: HAGI,
    optimizer: torch.optim.Optimizer,
    rl_dataset: list[dict[str, Any]],
    cfg: RLConfig,
    device: str = "cuda",
    start_step: int = 0,
    reward_fn: Any = None,
    eos_token_id: int | None = None,
    on_log: Any = None,
    on_checkpoint: Any = None,
) -> float:
    """Run the MGPO RL training loop.

    Args:
        model: HAGI model (loaded from SFT checkpoint).
        optimizer: optimizer for policy update.
        rl_dataset: list of {"prompt_ids": tensor, "reference": str} entries.
        cfg: RL configuration.
        device: target device.
        start_step: resume step.
        reward_fn: callable(responses: list[str], references: list[str]) -> Tensor.
        eos_token_id: EOS token id for generation stopping.
        on_log: optional callback for logging.
        on_checkpoint: optional callback for checkpointing.

    Returns:
        Final mean reward.
    """
    if reward_fn is None:
        from hagi.train.rewards import math_reward_fn
        reward_fn = math_reward_fn

    model.to(device)
    model.train()
    if hasattr(model.cfg, "gradient_checkpointing"):
        model.cfg.gradient_checkpointing = cfg.gradient_checkpointing

    torch.manual_seed(cfg.seed)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(cfg.seed)

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    last_reward = 0.0
    end = cfg.max_steps

    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    for step in range(start_step, end):
        lr = _lr_at(
            step, end, cfg.warmup_steps, cfg.learning_rate, cfg.min_lr_ratio
        )
        ratio = lr / max(cfg.learning_rate, 1e-12)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * ratio

        prompt_indices = torch.randint(
            0, len(rl_dataset), (cfg.num_prompts_per_step,), generator=rng
        )

        all_advantages: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        all_old_log_probs: list[torch.Tensor] = []
        total_reward = 0.0
        valid_groups = 0

        for pidx in prompt_indices:
            entry = rl_dataset[int(pidx.item())]
            prompt_ids = entry["prompt_ids"].to(device).unsqueeze(0)
            reference = entry["reference"]

            if prompt_ids.size(1) < cfg.min_prompt_len:
                continue

            rollouts: list[torch.Tensor] = []
            rollout_lengths: list[int] = []
            rollout_texts: list[str] = []

            from hagi.data.tokenizer import TokenizerWrapper

            tok = TokenizerWrapper.smollm2()

            for _g in range(cfg.group_size):
                full_ids, n_new = _generate_rollout(
                    model,
                    prompt_ids,
                    cfg.max_new_tokens,
                    cfg.temperature,
                    cfg.top_k,
                    cfg.top_p,
                    eos_token_id,
                )
                rollouts.append(full_ids)
                rollout_lengths.append(n_new)
                response_text = tok.decode(
                    full_ids[0, prompt_ids.size(1):].tolist()
                )
                rollout_texts.append(response_text)

            rewards = reward_fn(rollout_texts, [reference] * cfg.group_size)
            rewards = rewards.to(device)

            if not filter_by_difficulty(
                rewards, cfg.min_accuracy, cfg.max_accuracy
            ):
                continue

            if cfg.long2short_lambda > 0:
                lengths_tensor = torch.tensor(
                    rollout_lengths, dtype=torch.float32, device=device
                )
                rewards = long2short_reward_shift(
                    rewards, lengths_tensor, cfg.long2short_lambda
                )

            group_acc = float(rewards.mean().item())
            prompt_weight = mgpo_prompt_weight(
                group_acc, cfg.mgpo_gamma, cfg.mgpo_p0
            )

            advantages = group_relative_advantage(rewards) * prompt_weight
            total_reward += group_acc
            valid_groups += 1

            with torch.no_grad():
                for full_ids in rollouts:
                    resp_start = prompt_ids.size(1)
                    old_lp = _compute_log_probs(
                        model, full_ids, resp_start, cfg.precision
                    )
                    all_old_log_probs.append(old_lp)

            for full_ids in rollouts:
                resp_start = prompt_ids.size(1)
                lp = _compute_log_probs(
                    model, full_ids, resp_start, cfg.precision
                )
                all_log_probs.append(lp)

            for adv in advantages:
                all_advantages.append(adv)

        if valid_groups == 0:
            if cfg.log_interval > 0 and step % cfg.log_interval == 0:
                print(f"rl step {step}: no valid groups, skipping")
            continue

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        n_tokens = 0

        for lp, old_lp, adv in zip(
            all_log_probs, all_old_log_probs, all_advantages, strict=True
        ):
            min_len = min(lp.size(0), old_lp.size(0), 1)
            if min_len == 0:
                continue
            ratio_tokens = (lp[:min_len] - old_lp[:min_len]).exp()
            clipped = ratio_tokens.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps)
            adv_scalar = adv.clamp(-10, 10)
            surrogate = -torch.min(
                ratio_tokens * adv_scalar,
                clipped * adv_scalar,
            ).mean()
            total_loss = total_loss + surrogate
            n_tokens += 1

        if n_tokens > 0:
            loss = total_loss / n_tokens
            loss.backward()

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )

            optimizer.step()

        last_reward = total_reward / max(1, valid_groups)

        if cfg.log_interval > 0 and step % cfg.log_interval == 0:
            log_data = {
                "step": step,
                "lr": lr,
                "loss": float(total_loss.item() / max(1, n_tokens)),
                "mean_reward": last_reward,
                "valid_groups": valid_groups,
            }
            if on_log:
                on_log(log_data)
            else:
                print(
                    f"rl | step {step:6d} | lr {lr:.2e} | "
                    f"loss {log_data['loss']:.4f} | "
                    f"reward {last_reward:.3f} | "
                    f"groups {valid_groups}/{cfg.num_prompts_per_step}"
                )

        if cfg.ckpt_interval > 0 and step > 0 and step % cfg.ckpt_interval == 0:
            ckpt_path = ckpt_dir / f"step-{step:06d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "rl_config": cfg.__dict__,
                },
                ckpt_path,
            )
            if on_checkpoint:
                on_checkpoint(str(ckpt_path))
            print(f"checkpoint saved: {ckpt_path}")

    return last_reward
