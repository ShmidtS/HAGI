"""Verifiable reward functions for RL training (VibeThinker-inspired).

Implements:
- Math answer extraction + verification (final-answer matching)
- Long2Short reward redistribution (zero-sum brevity shift)
- MGPO prompt weighting (MaxEnt-guided boundary focus)

All rewards are binary {0, 1} for correctness, with optional length-aware
redistribution among correct trajectories.
"""

from __future__ import annotations

import re

import torch


_MATH_ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"),
    re.compile(r"(?:answer|Answer|ANSWER)\s*[:=]\s*(.+?)(?:\n|$)"),
    re.compile(r"(?:final answer|Final Answer)\s*[:=]\s*(.+?)(?:\n|$)"),
    re.compile(r"####\s*(.+?)(?:\n|$)"),
    re.compile(r"\[([^\[\]]+)\]\s*$"),
]

_CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_LAST_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def extract_math_answer(text: str) -> str:
    """Extract the final math answer from a generated response.

    Tries boxed, "answer:", "final answer:", "####", and [answer] patterns.
    Falls back to the last number in the text.
    """
    for pattern in _MATH_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
    numbers = _LAST_NUMBER_PATTERN.findall(text)
    if numbers:
        return numbers[-1]
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize a math answer for comparison: strip whitespace, commas, LaTeX."""
    s = answer.strip()
    s = s.replace("\\,", "").replace("\\ ", "").replace(" ", "")
    s = s.replace(",", "")
    s = s.rstrip(".")
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2]
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return f"{val:.6g}"
    except ValueError:
        return s.lower()


def math_verify(prediction: str, reference: str) -> bool:
    """Verify if the predicted answer matches the reference answer."""
    pred = normalize_answer(extract_math_answer(prediction))
    ref = normalize_answer(reference)
    if not pred or not ref:
        return False
    return pred == ref


def math_reward_fn(
    responses: list[str], references: list[str]
) -> torch.Tensor:
    """Compute binary math correctness rewards for a batch of responses.

    Args:
        responses: list of generated response strings.
        references: list of reference answer strings.

    Returns:
        Tensor of shape [B] with values in {0.0, 1.0}.
    """
    rewards = []
    for resp, ref in zip(responses, references, strict=True):
        rewards.append(1.0 if math_verify(resp, ref) else 0.0)
    return torch.tensor(rewards, dtype=torch.float32)


def long2short_reward_shift(
    rewards: torch.Tensor,
    response_lengths: torch.Tensor,
    lambda_brevity: float = 0.2,
) -> torch.Tensor:
    """Zero-sum length-aware reward redistribution (VibeThinker Long2Short).

    Among correct trajectories (reward=1), redistribute reward so shorter
    correct responses get more and longer ones get less. The sum of shifts
    is zero, so the group-level baseline is unchanged.

    Args:
        rewards: [G] binary correctness rewards.
        response_lengths: [G] token counts per response.
        lambda_brevity: max magnitude of the reward shift (default 0.2).

    Returns:
        [G] shifted rewards (incorrect unchanged, correct redistributed).
    """
    shifted = rewards.clone()
    correct_mask = rewards > 0.5
    if not correct_mask.any() or correct_mask.sum() == 1:
        return shifted

    correct_lengths = response_lengths[correct_mask].float()
    brevity = 1.0 / correct_lengths.clamp(min=1)
    brevity_mean = brevity.mean()
    brevity_diff = brevity - brevity_mean
    max_abs = brevity_diff.abs().max().clamp(min=1e-8)

    normalized = brevity_diff / max_abs
    shift = lambda_brevity * normalized

    correct_idx = correct_mask.nonzero(as_tuple=True)[0]
    shifted[correct_idx] = rewards[correct_idx] + shift
    return shifted


def mgpo_prompt_weight(
    group_accuracy: float, gamma: float = 4.0, p0: float = 0.5
) -> float:
    """Compute MGPO prompt weight based on group accuracy.

    w(q) = exp(-gamma * D_ME(p(q) || p0))

    Prompts near the capability boundary (p≈0.5) get the highest weight.
    Prompts that are too easy (p≈1) or too hard (p≈0) get suppressed.

    Args:
        group_accuracy: fraction of correct rollouts in the group.
        gamma: sharpness parameter (higher = more focused on boundary).
        p0: target accuracy point (0.5 = max entropy).

    Returns:
        Scalar weight in (0, 1].
    """
    p = max(0.0, min(1.0, group_accuracy))
    if p == 0.0 or p == 1.0:
        return 0.0
    entropy_diff = abs(p - p0)
    return float(torch.exp(torch.tensor(-gamma * entropy_diff)).item())


def group_relative_advantage(
    rewards: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Compute group-relative advantage (GRPO-style).

    A_i = (r_i - mean(r)) / (std(r) + eps)

    For binary rewards, this gives positive advantage to correct responses
    and negative to incorrect ones, normalized by group spread.

    Args:
        rewards: [G] reward tensor.
        eps: numerical stability.

    Returns:
        [G] advantage tensor.
    """
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


def filter_by_difficulty(
    rewards: torch.Tensor, min_accuracy: float = 0.0, max_accuracy: float = 1.0
) -> bool:
    """Check if a prompt group should be kept for RL training.

    VibeThinker filters out prompts with accuracy exactly 0.0 (too hard)
    or 1.0 (too easy) before training. This implementation allows configurable
    thresholds.

    Args:
        rewards: [G] binary rewards for the group.
        min_accuracy: minimum group accuracy to keep (exclusive).
        max_accuracy: maximum group accuracy to keep (exclusive).

    Returns:
        True if the group should be kept for training.
    """
    accuracy = float(rewards.mean().item())
    return min_accuracy < accuracy < max_accuracy


def compute_clr_reliability(
    claim_verdicts: torch.Tensor,
) -> torch.Tensor:
    """Claim-Level Reliability score (VibeThinker CLR).

    r_k = (1/M * sum(v_{k,m}))^M

    Nonlinear mapping that heavily penalizes trajectories with any
    failed claim verification.

    Args:
        claim_verdicts: [K, M] binary verdicts (1=valid, 0=falsified).

    Returns:
        [K] reliability scores in [0, 1].
    """
    k, m = claim_verdicts.shape
    if m == 0:
        return torch.ones(k, dtype=torch.float32)
    mean_valid = claim_verdicts.float().mean(dim=1)
    return mean_valid.pow(m)
