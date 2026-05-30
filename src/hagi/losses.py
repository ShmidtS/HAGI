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


def compute_auxiliary_loss(gdr_output: torch.Tensor) -> torch.Tensor:
    """Penalize similarity between grade projections."""
    flattened = gdr_output.reshape(-1, gdr_output.size(-1)).float()
    if flattened.size(0) < 2:
        return gdr_output.new_zeros(())
    normalized = F.normalize(flattened, dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    off_diagonal = ~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    return similarity[off_diagonal].pow(2).mean()


def _as_logits(output: torch.Tensor | tuple | dict) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        return output["logits"]
    raise TypeError("model output must be a tensor, tuple, or dict")


def compute_isomorphic_loss(model, input_ids: torch.Tensor, targets: torch.Tensor, device) -> torch.Tensor:
    """Run two forward passes and penalize output differences."""
    input_ids = input_ids.to(device)
    first = _as_logits(model(input_ids))
    second = _as_logits(model(input_ids))
    return F.mse_loss(first.float(), second.float())


def composite_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    model_output: torch.Tensor,
    auxiliary_output: torch.Tensor | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute CE, auxiliary, isomorphic, and weighted total losses."""
    merged_weights = {"w_ce": 1.0, "w_aux": 0.1, "w_iso": 0.01}
    if weights is not None:
        merged_weights.update(weights)

    l_ce = cross_entropy_loss(logits.float(), targets)
    l_aux = logits.new_zeros(()) if auxiliary_output is None else compute_auxiliary_loss(auxiliary_output)
    l_iso = isomorphic_consistency_loss(logits.float(), model_output.float())
    l_total = (
        merged_weights["w_ce"] * l_ce
        + merged_weights["w_aux"] * l_aux
        + merged_weights["w_iso"] * l_iso
    )
    return {"L_CE": l_ce, "L_aux": l_aux, "L_iso": l_iso, "L_total": l_total}
