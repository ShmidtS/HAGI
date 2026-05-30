"""Training loss helpers for HAGI."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute token cross-entropy with class logits in the final dimension."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def auxiliary_gdr_loss(
    gdr_output: torch.Tensor,
    grade_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Placeholder auxiliary GDR loss for grade separation targets."""
    if grade_targets is None:
        return gdr_output.new_zeros(())
    return F.mse_loss(gdr_output, grade_targets)


def isomorphic_consistency_loss(
    model_output: torch.Tensor,
    target_output: torch.Tensor,
) -> torch.Tensor:
    """Penalize differences between two forward-pass outputs."""
    return F.mse_loss(model_output, target_output)


def total_loss(
    losses_dict: Mapping[str, torch.Tensor],
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Combine component losses with optional scalar weights."""
    if not losses_dict:
        return torch.tensor(0.0)

    total: torch.Tensor | None = None
    for name, loss in losses_dict.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        component = loss * weight
        total = component if total is None else total + component

    assert total is not None
    return total
