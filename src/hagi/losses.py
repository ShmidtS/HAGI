"""Training loss helpers for HAGI."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def _cross_entropy_impl(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    chunk_size: int = 0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute token cross-entropy with class logits in the final dimension.

    logits: [..., V] (any shape; last dim is vocab). Internally flattened to [N, V].
    The fp32 upcast of the full [N, V] tensor is the dominant activation-memory
    spike at large N·V (e.g. 8·1024·49152·4B ≈ 1.6 GB). When chunk_size > 0, the
    upcast happens per row-chunk so the fp32 copy never fully materializes.
    Numerically identical to the unchunked path (sum over chunks / valid-token
    count == mean). ``label_smoothing`` in [0,1) is forwarded to
    ``F.cross_entropy`` (improves small-model generalization).
    """
    if logits.ndim > 2:
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
    if chunk_size <= 0 or logits.size(0) <= chunk_size:
        return F.cross_entropy(
            logits, targets, ignore_index=ignore_index, label_smoothing=label_smoothing
        )
    valid = (targets != ignore_index).sum().clamp(min=1)
    # fp32 accumulator: summing chunk losses in bf16 loses precision and the
    # native CE kernel already returns fp32 for bf16 input, so this matches.
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    for i in range(0, logits.size(0), chunk_size):
        lg = logits[i : i + chunk_size]
        tg = targets[i : i + chunk_size]
        total = total + F.cross_entropy(
            lg,
            tg,
            ignore_index=ignore_index,
            reduction="sum",
            label_smoothing=label_smoothing,
        )
    return total / valid


_cross_entropy_compiled = (
    torch.compile(_cross_entropy_impl, mode="default", dynamic=False)
    if torch.cuda.is_available()
    else None
)


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    chunk_size: int = 0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    fn = (
        _cross_entropy_compiled
        if logits.is_cuda and _cross_entropy_compiled is not None
        else _cross_entropy_impl
    )
    return fn(logits, targets, ignore_index, chunk_size, label_smoothing)


@torch.compiler.disable
def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    chunk_size: int = 4096,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Chunked lm_head projection + cross-entropy without materializing [N, V] logits.

    Each row-chunk is projected and reduced under activation checkpointing, so at
    most ``chunk_size x V`` logits exist at any time (in forward AND backward; the
    chunk is recomputed during backward). With V = 32k and N = B*T the full logits
    tensor is the dominant activation-memory term; this path cuts it by a factor
    of N / chunk_size. Numerically identical to the unchunked path (sum over
    chunks / valid-token count == mean).

    ``@torch.compiler.disable``: this is called from inside the compiled model
    graph (hagi.py:647), and the chunk loop ``for i in range(0, flat_h.size(0),
    chunk_size)`` has a data-dependent trip count (variable-length training ->
    B*T varies -> ceil(B*T/chunk_size) is 1..3). Dynamo specializes a guard on
    the trip count and hits recompile_limit(8) -> eager fallback, wasting the
    step-0 compile. Disabling keeps the eager chunk loop (it is just
    F.linear + F.cross_entropy on a [chunk, V] slab — the CUDA kernels are
    already optimal, compile adds nothing) and stops the guard. Mirrors the
    SlotRegistry disable pattern (msa.py).
    """
    flat_h = hidden.reshape(-1, hidden.size(-1))
    flat_t = targets.reshape(-1)
    if chunk_size <= 0:
        chunk_size = 4096
    valid = (flat_t != ignore_index).sum().clamp(min=1)

    def _chunk_loss(h_chunk: torch.Tensor, t_chunk: torch.Tensor) -> torch.Tensor:
        logits = F.linear(h_chunk, weight)
        # No explicit .float(): the native CUDA CE kernel accumulates in fp32
        # internally for bf16/fp16 input, so upcasting the full [chunk, V]
        # logits is pure bandwidth overhead (halves the chunk's memory spike).
        return F.cross_entropy(
            logits,
            t_chunk,
            ignore_index=ignore_index,
            reduction="sum",
            label_smoothing=label_smoothing,
        )

    # Chunking (not checkpointing) bounds peak memory: only one [chunk, V]
    # logits tensor is alive at a time. Checkpointing the chunk would recompute
    # the lm_head matmul + log_softmax on backward — ~30% slower here for no
    # memory benefit, since the chunk is already small (chunk_size * V * 2B).
    total = flat_h.new_zeros((), dtype=torch.float32)
    for i in range(0, flat_h.size(0), chunk_size):
        h_c = flat_h[i : i + chunk_size]
        t_c = flat_t[i : i + chunk_size]
        total = total + _chunk_loss(h_c, t_c)
    return (total / valid).to(hidden.dtype)


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
    else:
        labels = labels.to(device=features.device)
    labels = labels.reshape(-1)
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

    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12).log()
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    loss = -(log_prob * positive_mask).sum(dim=1)
    loss = (loss / positive_count.clamp_min(1)) * valid.float()
    return loss.sum() / valid.sum().clamp_min(1)


def compute_isomorphic_loss(
    invariant_src,
    invariant_tgt=None,
) -> torch.Tensor:
    """Compute invariant MSE when HDIM source and target invariants are available."""
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

    if isinstance(invariant_src, torch.Tensor) and isinstance(
        invariant_tgt, torch.Tensor
    ):
        return F.mse_loss(invariant_src, invariant_tgt)
    if isinstance(invariant_src, torch.Tensor):
        return invariant_src.new_zeros(())
    return torch.tensor(0.0)


def composite_loss(
    logits: torch.Tensor | None,
    targets: torch.Tensor,
    auxiliary_output=None,
    model_output: torch.Tensor | dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    invariant_src=None,
    invariant_tgt=None,
    precomputed_loss: torch.Tensor | None = None,
    moe_aux_loss: torch.Tensor | None = None,
    num_moe_layers: int | torch.Tensor | None = None,
    msa_aux_loss: torch.Tensor | None = None,
    chunk_size: int = 0,
    quality_score: torch.Tensor | None = None,
    quality_targets: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute CE, auxiliary, isomorphic, and weighted total losses."""
    # Pick a reference tensor for device/dtype when logits is None (fused CE path)
    ref_tensor = (
        logits
        if logits is not None
        else (precomputed_loss if precomputed_loss is not None else targets)
    )

    if (
        isinstance(auxiliary_output, torch.Tensor)
        and isinstance(model_output, torch.Tensor)
        and logits is not None
        and auxiliary_output.shape == logits.shape
        and model_output.shape != logits.shape
    ):
        auxiliary_output, model_output = model_output, auxiliary_output

    if invariant_src is None and isinstance(model_output, dict):
        invariant_src = model_output.get("invariant_src")
    if invariant_tgt is None and isinstance(model_output, dict):
        invariant_tgt = model_output.get("invariant_tgt")
    # Fallback: if model_output is None but auxiliary_output is a dict with these keys
    if invariant_src is None and isinstance(auxiliary_output, dict):
        invariant_src = auxiliary_output.get("invariant_src")
    if invariant_tgt is None and isinstance(auxiliary_output, dict):
        invariant_tgt = auxiliary_output.get("invariant_tgt")
    if moe_aux_loss is None and isinstance(auxiliary_output, dict):
        moe_aux_loss = auxiliary_output.get("moe_aux_loss")
    if num_moe_layers is None and isinstance(auxiliary_output, dict):
        num_moe_layers = auxiliary_output.get("num_moe_layers")
    if msa_aux_loss is None and isinstance(auxiliary_output, dict):
        msa_aux_loss = auxiliary_output.get("msa_aux_loss")
    if msa_aux_loss is None and isinstance(model_output, dict):
        msa_aux_loss = model_output.get("msa_aux_loss")

    merged_weights = {
        "w_ce": 1.0,
        "w_aux": 0.1,
        "w_iso": 0.01,
        "w_moe": 0.0,
        "w_msa_lb": 0.0,
        "w_quality": 0.0,
    }
    if weights is not None:
        merged_weights.update(weights)

    if precomputed_loss is not None:
        l_ce = precomputed_loss
    else:
        if logits is None:
            raise ValueError("logits is None and precomputed_loss is not provided")
        l_ce = cross_entropy_loss(logits, targets, chunk_size=chunk_size)
    l_total = merged_weights["w_ce"] * l_ce
    result = {"L_CE": l_ce}
    if merged_weights.get("w_aux", 0.0) != 0.0 and auxiliary_output is not None:
        # If auxiliary_output is a dict containing nested "auxiliary_output", unwrap it
        aux_payload = auxiliary_output
        if (
            isinstance(auxiliary_output, dict)
            and "auxiliary_output" in auxiliary_output
        ):
            aux_payload = auxiliary_output["auxiliary_output"]
        l_aux = compute_auxiliary_loss(aux_payload).to(
            device=ref_tensor.device, dtype=ref_tensor.dtype
        )
        l_total = l_total + merged_weights["w_aux"] * l_aux
        result["L_aux"] = l_aux
    else:
        result["L_aux"] = l_ce.new_zeros(())
    if merged_weights.get("w_iso", 0.0) != 0.0 and (
        invariant_src is not None or invariant_tgt is not None
    ):
        l_iso = compute_isomorphic_loss(invariant_src, invariant_tgt).to(
            device=ref_tensor.device, dtype=ref_tensor.dtype
        )
        l_total = l_total + merged_weights["w_iso"] * l_iso
        result["L_iso"] = l_iso
    else:
        result["L_iso"] = l_ce.new_zeros(())
    if merged_weights.get("w_moe", 0.0) != 0.0 and moe_aux_loss is not None:
        l_moe = moe_aux_loss.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        if num_moe_layers is not None:
            if isinstance(num_moe_layers, torch.Tensor):
                num_moe_layers = int(num_moe_layers.item())
            if num_moe_layers > 0:
                l_moe = l_moe / num_moe_layers
        l_total = l_total + merged_weights["w_moe"] * l_moe
        result["L_moe"] = l_moe
    else:
        result["L_moe"] = l_ce.new_zeros(())
    if merged_weights.get("w_msa_lb", 0.0) != 0.0 and msa_aux_loss is not None:
        # MSA router load-balance: no num_msa_layers (single MSA block per fwd).
        l_msa_lb = msa_aux_loss.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        l_total = l_total + merged_weights["w_msa_lb"] * l_msa_lb
        result["L_msa_lb"] = l_msa_lb
    else:
        result["L_msa_lb"] = l_ce.new_zeros(())
    if (
        merged_weights.get("w_quality", 0.0) != 0.0
        and quality_score is not None
        and quality_targets is not None
    ):
        valid = quality_targets != -1.0
        if valid.any():
            l_quality = F.binary_cross_entropy_with_logits(
                quality_score[valid], quality_targets[valid], reduction="mean"
            ).to(device=ref_tensor.device, dtype=ref_tensor.dtype)
            l_total = l_total + merged_weights["w_quality"] * l_quality
            result["L_quality"] = l_quality
        else:
            result["L_quality"] = l_ce.new_zeros(())
    else:
        result["L_quality"] = l_ce.new_zeros(())
    result["L_total"] = l_total
    return result
