"""Training loss helpers for HAGI."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


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


def compute_auxiliary_loss(aux_output, max_samples: int = 256) -> torch.Tensor:
    """Compute supervised contrastive auxiliary loss when pair labels are available.

    Subsamples to ``max_samples`` tokens to keep the O(N^2) similarity matrix bounded.
    """
    if aux_output is None:
        return torch.tensor(0.0)

    labels = None
    features = aux_output
    if isinstance(aux_output, dict):
        for key in ("features", "embeddings", "output"):
            value = aux_output.get(key)
            if isinstance(value, torch.Tensor):
                features = value
                break
        for key in ("labels", "pair_labels"):
            value = aux_output.get(key)
            if value is not None:
                labels = value
                break
    elif isinstance(aux_output, tuple):
        if len(aux_output) >= 1:
            features = aux_output[0]
        if len(aux_output) >= 2:
            labels = aux_output[1]

    if not isinstance(features, torch.Tensor):
        return torch.tensor(0.0)
    flat = features.reshape(-1, features.size(-1))
    if labels is None:
        logger.debug("auxiliary contrastive labels missing; L_aux set to 0")
        return flat.new_zeros(())
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, device=features.device)

    flat = features.reshape(-1, features.size(-1))
    labels = labels.to(device=features.device).reshape(-1)
    n = flat.size(0)
    if n != labels.numel() or n < 2:
        logger.debug("auxiliary contrastive labels invalid; L_aux set to 0")
        return flat.new_zeros(())

    # Subsample if too many tokens
    if n > max_samples:
        idx = torch.randperm(n, device=flat.device)[:max_samples]
        flat = flat[idx]
        labels = labels[idx]

    flat_norm = F.normalize(flat, dim=-1)
    logits = torch.mm(flat_norm, flat_norm.t()) / 0.07
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    if not positive_mask.any():
        logger.debug("auxiliary contrastive positive pairs missing; L_aux set to 0")
        return flat.new_zeros(())

    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12).log()
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    return -(log_prob * positive_mask).sum(dim=1)[valid].div(positive_count[valid]).mean()


def _as_logits(output: torch.Tensor | tuple[Any, ...] | dict[str, Any]) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        return output["logits"]
    raise TypeError("model output must be a tensor, tuple, or dict")


def _model_output_tensor(model_output: torch.Tensor | dict[str, Any] | None) -> torch.Tensor | None:
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
    invariant_src,
    invariant_tgt=None,
    targets: torch.Tensor | None = None,
    device=None,
) -> torch.Tensor:
    """Compute invariant MSE when HDIM source and target invariants are available."""
    if isinstance(invariant_src, torch.nn.Module):
        if invariant_tgt is None or device is None:
            raise ValueError("input_ids and device are required when passing a model")
        input_ids = invariant_tgt.to(device)
        first = _as_logits(invariant_src(input_ids))
        second = _as_logits(invariant_src(input_ids))
        return F.mse_loss(first, second)

    if isinstance(invariant_src, dict):
        src = None
        tgt = None
        for key in ("invariant_src", "invariant"):
            value = invariant_src.get(key)
            if isinstance(value, torch.Tensor):
                src = value
                break
        for key in ("invariant_tgt", "target_invariant"):
            value = invariant_src.get(key)
            if isinstance(value, torch.Tensor):
                tgt = value
                break
        if isinstance(src, torch.Tensor) and isinstance(tgt, torch.Tensor):
            return F.mse_loss(src, tgt)
        return torch.tensor(0.0)

    if isinstance(invariant_src, torch.Tensor) and isinstance(invariant_tgt, torch.Tensor):
        return F.mse_loss(invariant_src, invariant_tgt)
    if isinstance(invariant_src, torch.Tensor):
        return invariant_src.new_zeros(())
    return torch.tensor(0.0)


def composite_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    auxiliary_output=None,
    model_output: torch.Tensor | dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    invariant_src=None,
    invariant_tgt=None,
    precomputed_loss: torch.Tensor | None = None,
    moe_aux_loss: torch.Tensor | None = None,
    num_moe_layers: int | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute CE, auxiliary, isomorphic, and weighted total losses."""
    if (
        isinstance(auxiliary_output, torch.Tensor)
        and isinstance(model_output, torch.Tensor)
        and auxiliary_output.shape == logits.shape
        and model_output.shape != logits.shape
    ):
        auxiliary_output, model_output = model_output, auxiliary_output

    if invariant_src is None and isinstance(model_output, dict):
        invariant_src = model_output.get("invariant_src")
    if invariant_tgt is None and isinstance(model_output, dict):
        invariant_tgt = model_output.get("invariant_tgt")

    merged_weights = {"w_ce": 1.0, "w_aux": 0.1, "w_iso": 0.01, "w_moe": 0.0}
    if weights is not None:
        merged_weights.update(weights)

    if precomputed_loss is not None:
        l_ce = precomputed_loss
    else:
        l_ce = cross_entropy_loss(logits, targets)
    l_total = merged_weights["w_ce"] * l_ce
    result = {"L_CE": l_ce}
    if merged_weights.get("w_aux", 0.0) != 0.0 and auxiliary_output is not None:
        l_aux = compute_auxiliary_loss(auxiliary_output).to(device=logits.device, dtype=logits.dtype)
        l_total = l_total + merged_weights["w_aux"] * l_aux
        result["L_aux"] = l_aux
    else:
        result["L_aux"] = l_ce.new_zeros(())
    if merged_weights.get("w_iso", 0.0) != 0.0 and (invariant_src is not None or invariant_tgt is not None):
        l_iso = compute_isomorphic_loss(invariant_src, invariant_tgt).to(device=logits.device, dtype=logits.dtype)
        l_total = l_total + merged_weights["w_iso"] * l_iso
        result["L_iso"] = l_iso
    else:
        result["L_iso"] = l_ce.new_zeros(())
    if merged_weights.get("w_moe", 0.0) != 0.0 and moe_aux_loss is not None:
        l_moe = moe_aux_loss.to(device=logits.device, dtype=logits.dtype)
        if num_moe_layers is not None:
            if isinstance(num_moe_layers, torch.Tensor):
                num_moe_layers = int(num_moe_layers.item())
            if num_moe_layers > 0:
                l_moe = l_moe / num_moe_layers
        l_total = l_total + merged_weights["w_moe"] * l_moe
        result["L_moe"] = l_moe
    else:
        result["L_moe"] = l_ce.new_zeros(())
    result["L_total"] = l_total
    return result
