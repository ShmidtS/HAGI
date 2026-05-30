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


def compute_auxiliary_loss(gdr_output: torch.Tensor | None) -> torch.Tensor:
    """Penalize similarity between grade projections."""
    if gdr_output is None:
        return torch.tensor(0.0)
    flat = gdr_output.float().reshape(-1, gdr_output.size(-1))
    if flat.size(0) < 2 or flat.size(-1) < 1:
        return flat.new_zeros(())
    flat_norm = F.normalize(flat, dim=-1)
    sim = torch.mm(flat_norm, flat_norm.t())
    mask = 1.0 - torch.eye(sim.size(0), dtype=sim.dtype, device=sim.device)
    return (sim * mask).abs().mean()


def _as_logits(output: torch.Tensor | tuple | dict) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        return output["logits"]
    raise TypeError("model output must be a tensor, tuple, or dict")


def _model_output_tensor(model_output: torch.Tensor | dict | None) -> torch.Tensor | None:
    if model_output is None:
        return None
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, dict):
        for key in ("hidden_states", "output", "logits"):
            value = model_output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    return None


def compute_isomorphic_loss(
    model_output,
    input_ids: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    device=None,
) -> torch.Tensor:
    """Penalize nondeterministic or unbounded model outputs."""
    if isinstance(model_output, torch.nn.Module):
        if input_ids is None or device is None:
            raise ValueError("input_ids and device are required when passing a model")
        input_ids = input_ids.to(device)
        first = _as_logits(model_output(input_ids))
        second = _as_logits(model_output(input_ids))
        return F.mse_loss(first.float(), second.float())

    if isinstance(model_output, dict):
        first = _model_output_tensor(model_output)
        second = model_output.get("second_forward")
        if second is None:
            second = model_output.get("reference")
        if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
            return F.mse_loss(first.float(), second.float())
        if isinstance(first, torch.Tensor):
            return first.float().pow(2).mean()
        return torch.tensor(0.0)

    tensor = _model_output_tensor(model_output)
    if tensor is None:
        return torch.tensor(0.0)
    return tensor.float().pow(2).mean()


def composite_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    auxiliary_output: torch.Tensor | None = None,
    model_output: torch.Tensor | dict | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute CE, auxiliary, isomorphic, and weighted total losses."""
    if (
        isinstance(auxiliary_output, torch.Tensor)
        and isinstance(model_output, torch.Tensor)
        and auxiliary_output.shape == logits.shape
        and model_output.shape != logits.shape
    ):
        auxiliary_output, model_output = model_output, auxiliary_output

    merged_weights = {"w_ce": 1.0, "w_aux": 0.1, "w_iso": 0.01}
    if weights is not None:
        merged_weights.update(weights)

    l_ce = cross_entropy_loss(logits.float(), targets)
    l_aux = compute_auxiliary_loss(auxiliary_output).to(device=logits.device, dtype=logits.dtype)
    l_iso = compute_isomorphic_loss(model_output).to(device=logits.device, dtype=logits.dtype)
    l_total = (
        merged_weights["w_ce"] * l_ce
        + merged_weights["w_aux"] * l_aux
        + merged_weights["w_iso"] * l_iso
    )
    return {"L_CE": l_ce, "L_aux": l_aux, "L_iso": l_iso, "L_total": l_total}
