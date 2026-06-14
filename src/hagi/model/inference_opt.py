"""Inference-only optimizations for HAGI.

These transforms are NOT safe to apply to a model that will continue training:
- RMSNorm folding destroys gradient flow through the norm layer.
- Weight repacking changes the parameter layout.
- RoPE precompute is safe but ties the model to a max_seq_len.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from .transformer import (
    GroupedQueryAttention,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    build_rope_cache,
)


def fold_rmsnorm_into_weights(model: nn.Module) -> nn.Module:
    """Fold RMSNorm scaling into adjacent linear weights.

    For each TransformerBlock:
    - attn_norm weight is folded into attn.q_proj, attn.k_proj, attn.v_proj
    - mlp_norm weight is folded into mlp.gate, mlp.up

    After folding, norm layers are replaced with nn.Identity so the inference
    path skips them entirely.

    final_norm is intentionally NOT folded into lm_head because lm_head.weight
    is tied to embed.weight in HAGI; folding would corrupt the embedding table.
    """
    folded_any = False
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            if isinstance(module.attn_norm, RMSNorm):
                assert isinstance(module.attn_norm, RMSNorm)
                gamma = module.attn_norm.weight.data
                if isinstance(module.attn, GroupedQueryAttention) and hasattr(
                    module.attn, "qkv_weight"
                ):
                    qkv = module.attn.qkv_weight
                    q, k, v = qkv.split(module.attn._qkv_splits, dim=0)
                    q = q * gamma.view(1, -1)
                    k = k * gamma.view(1, -1)
                    v = v * gamma.view(1, -1)
                    qkv.data = torch.cat([q, k, v], dim=0).contiguous()
                else:
                    for proj in (
                        module.attn.q_proj,
                        module.attn.k_proj,
                        module.attn.v_proj,
                    ):
                        if isinstance(proj, nn.Linear):
                            with torch.no_grad():
                                proj.weight.data.mul_(gamma.view(1, -1))  # type: ignore
                object.__setattr__(module, "_attn_norm_eps", module.attn_norm.eps)
                object.__setattr__(module, "attn_norm", nn.Identity())
                folded_any = True

            if isinstance(module.mlp_norm, RMSNorm):
                assert isinstance(module.mlp_norm, RMSNorm)
                gamma = module.mlp_norm.weight.data
                if hasattr(module.mlp, "gate_up_weight"):
                    assert isinstance(module.mlp.gate_up_weight, torch.Tensor)
                    gate, up = module.mlp.gate_up_weight.chunk(2, dim=0)
                    gate = gate * gamma.view(1, -1)
                    up = up * gamma.view(1, -1)
                    module.mlp.gate_up_weight.data = torch.cat(
                        [gate, up], dim=0
                    ).contiguous()
                elif hasattr(module.mlp, "experts"):
                    # Mixture-of-Experts SwiGLU: fold gamma into each expert's
                    # gate/up projections (the first ops after mlp_norm).
                    experts = cast(Any, module.mlp).experts
                    with torch.no_grad():
                        for expert in experts:
                            for proj in (expert.gate, expert.up):
                                if isinstance(proj, nn.Linear):
                                    proj.weight.data.mul_(gamma.view(1, -1))  # type: ignore
                else:
                    for proj in (module.mlp.gate, module.mlp.up):
                        if isinstance(proj, nn.Linear):
                            with torch.no_grad():
                                proj.weight.data.mul_(gamma.view(1, -1))  # type: ignore
                object.__setattr__(module, "_mlp_norm_eps", module.mlp_norm.eps)
                object.__setattr__(module, "mlp_norm", nn.Identity())
                folded_any = True

    # Fold final_norm into lm_head ONLY if weights are NOT tied.
    if (
        hasattr(model, "lm_head")
        and hasattr(model, "final_norm")
        and hasattr(model, "embed")
    ):
        lm_head_obj = model.lm_head
        embed_obj = model.embed
        assert isinstance(lm_head_obj, nn.Linear)
        assert isinstance(embed_obj, nn.Embedding)
        if (
            isinstance(model.final_norm, RMSNorm)
            and lm_head_obj.weight is not embed_obj.weight
        ):
            gamma = model.final_norm.weight.data  # type: ignore
            with torch.no_grad():
                lm_head_obj.weight.data.mul_(gamma.view(1, -1))  # type: ignore
            model.final_norm = cast(Any, nn.Identity())
            folded_any = True

    if not folded_any:
        raise RuntimeError(
            "fold_rmsnorm_into_weights: no RMSNorm layers found to fold."
        )
    return model


def repack_qkv_for_contiguous(model: nn.Module) -> nn.Module:
    """Repack separate Q/K/V and gate/up weights into contiguous tensors.

    Adds ``qkv_weight`` buffer to GroupedQueryAttention and
    ``gate_up_weight`` buffer to SwiGLU.  The repacked forward paths
    (``forward_repacked``) are automatically selected when
    ``self.training is False``.
    """
    repacked_any = False
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            attn = module.attn
            if isinstance(attn, GroupedQueryAttention):
                if not hasattr(attn, "qkv_weight"):
                    if (
                        hasattr(attn, "q_proj")
                        and hasattr(attn, "k_proj")
                        and hasattr(attn, "v_proj")
                    ):
                        if (
                            isinstance(attn.q_proj, nn.Linear)
                            and isinstance(attn.k_proj, nn.Linear)
                            and isinstance(attn.v_proj, nn.Linear)
                        ):
                            wq = attn.q_proj.weight.data
                            wk = attn.k_proj.weight.data
                            wv = attn.v_proj.weight.data
                            qkv = torch.cat([wq, wk, wv], dim=0).contiguous()
                            attn.register_buffer("qkv_weight", qkv)
                            object.__setattr__(
                                attn,
                                "_qkv_splits",
                                [wq.size(0), wk.size(0), wv.size(0)],
                            )
                            repacked_any = True
                else:
                    repacked_any = True

            mlp = module.mlp
            if isinstance(mlp, SwiGLU):
                if not hasattr(mlp, "gate_up_weight"):
                    if hasattr(mlp, "gate") and hasattr(mlp, "up"):
                        if isinstance(mlp.gate, nn.Linear) and isinstance(
                            mlp.up, nn.Linear
                        ):
                            w1 = mlp.gate.weight.data
                            w3 = mlp.up.weight.data
                            gate_up = torch.cat([w1, w3], dim=0).contiguous()
                            mlp.register_buffer("gate_up_weight", gate_up)
                            repacked_any = True
                else:
                    repacked_any = True

    if not repacked_any:
        raise RuntimeError(
            "repack_qkv_for_contiguous: no eligible attention/MLP blocks found."
        )
    return model


def precompute_rope_tables(model: nn.Module, max_seq_len: int) -> nn.Module:
    """Precompute RoPE cos/sin tables up to ``max_seq_len`` and store as buffers.

    After precompute, ``HAGI._rope_cache`` returns slices from the buffers
    instead of calling ``build_rope_cache`` on every forward pass.
    """
    from .hagi import HAGI

    if not isinstance(model, HAGI):
        raise TypeError("precompute_rope_tables expects a HAGI model")

    head_dim = (
        model.cfg.transformer.hidden_size // model.cfg.transformer.num_query_heads
    )
    cos, sin = build_rope_cache(
        max_seq_len,
        head_dim,
        model.cfg.transformer.rope_theta,
        torch.device("cpu"),
        torch.float32,
    )
    model.register_buffer("_rope_cos", cos)
    model.register_buffer("_rope_sin", sin)
    object.__setattr__(model, "_rope_max_seq_len", max_seq_len)
    return model


def pin_model_weights(model: nn.Module) -> nn.Module:
    """Ensure all parameters are on GPU and contiguous for inference.

    This is useful before a tight inference loop to avoid hidden host-to-device
    copies or non-contiguous strided accesses.
    """
    for p in model.parameters():
        if not p.is_cuda:
            p.data = p.data.cuda(non_blocking=True)
        if not p.is_contiguous():
            p.data = p.data.contiguous()
    for b in model.buffers():
        if not b.is_cuda:
            b.data = b.data.cuda(non_blocking=True)
        if not b.is_contiguous():
            b.data = b.data.contiguous()
    return model
