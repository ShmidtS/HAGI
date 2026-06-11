"""HAGI model — Perception / Reasoning / Expression with optional GDR and MSA.

A single class covers all four ablation models via config flags:

    Model A (baseline): use_loop=False, use_gdr=False
    Model B (loop):     use_loop=True,  use_gdr=False
    Model C (HDIM):     use_loop=False, use_gdr=True   (Clifford bolted on, no loop)
    Model D (GDR):      use_loop=True,  use_gdr=True   (full HAGI)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..losses import cross_entropy_loss, fused_linear_cross_entropy
from ..nars.adapters import NarsHdimReasoner, NarsHrmController, NarsMsaReasoner
from .gdr import GradeConfig, GradeDecomposedRecurrence
from .hdim_full import DelayedHDIM, HDIMFull
from .hrm_full import HRMCore
from .msa import HDIMSlotRouter, MSAAttention, SlotRegistry, SparseRouter
from .transformer import RMSNorm, TransformerBlock, TransformerConfig, build_rope_cache


def _pick_rotor_idx(seed: int, step: int, num_rotors: int) -> int:
    """Pick a target rotor index via LCG (no GPU sync, no object allocation)."""
    if num_rotors <= 1:
        return 0
    state = (seed * 1103515245 + step * 12345) & 0x7FFFFFFF
    return (state % (num_rotors - 1)) + 1


@dataclass
class HAGIConfig:
    vocab_size: int = 49152
    hidden_size: int = 768
    perception_layers: int = 4
    reasoning_layers: int = 4
    expression_layers: int = 4
    loop_count: int = 1
    use_loop: bool = True
    use_gdr: bool = True
    hdim_full: bool = True
    hdim_heads: int = 4
    hdim_delay_steps: int = 1
    hrm: bool = True
    hrm_h_cycles: int = 1
    hrm_l_cycles: int = 3
    h_dim: int = 256
    l_dim: int = 256
    gradient_checkpointing: bool = True
    rotor_seed: int = 42
    use_hdim_cross_domain: bool = True
    use_msa: bool = True
    msa_slot_count: int = 100
    msa_top_k: int = 5
    use_nars: bool = True
    thinking_noise: float = 0.0
    use_quality_head: bool = False
    use_binary_factorized: bool = False
    binary_factorized_rank: int = 8
    use_moe: bool = True
    num_experts: int = 4
    moe_top_k: int = 1
    moe_intermediate_size: int | None = None
    moe_alpha: float = 0.01
    ce_chunk_size: int = 0
    use_fused_ce: bool = False
    compile: bool = False
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    grades: GradeConfig = field(default_factory=GradeConfig)

    def __post_init__(self):
        assert self.hidden_size == self.transformer.hidden_size
        self.transformer.use_binary_factorized = self.use_binary_factorized
        self.transformer.binary_factorized_rank = self.binary_factorized_rank
        self.transformer.use_moe = self.use_moe
        self.transformer.num_experts = self.num_experts
        self.transformer.moe_top_k = self.moe_top_k
        self.transformer.moe_intermediate_size = self.moe_intermediate_size
        self.transformer.moe_alpha = self.moe_alpha
        if self.use_moe and self.transformer.moe_intermediate_size is None:
            self.transformer.moe_intermediate_size = (
                self.transformer.intermediate_size // self.num_experts
            )
        if self.use_gdr and not self.hdim_full and not self.hrm:
            assert (
                self.hidden_size == self.grades.hidden_size
            ), f"grade dims sum to {self.grades.hidden_size}, hidden is {self.hidden_size}"


class HAGI(nn.Module):
    _step: torch.Tensor

    def __init__(self, cfg: HAGIConfig):
        super().__init__()
        self.cfg = cfg
        tcfg = cfg.transformer

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.perception = nn.ModuleList(
            TransformerBlock(tcfg) for _ in range(cfg.perception_layers)
        )
        self.reasoning = nn.ModuleList(
            TransformerBlock(tcfg) for _ in range(cfg.reasoning_layers)
        )
        self.expression = nn.ModuleList(
            TransformerBlock(tcfg) for _ in range(cfg.expression_layers)
        )

        self.gdr = None
        if cfg.use_gdr:
            if cfg.hdim_full:
                hdim_module = (
                    DelayedHDIM(
                        hidden_size=cfg.hidden_size,
                        heads=cfg.hdim_heads,
                        delay_steps=cfg.hdim_delay_steps,
                        grades=cfg.grades,
                    )
                    if cfg.hdim_delay_steps > 1
                    else HDIMFull(
                        hidden_size=cfg.hidden_size,
                        heads=cfg.hdim_heads,
                        grades=cfg.grades,
                    )
                )
                if not getattr(cfg, "use_hdim_cross_domain", False):
                    hdim_module.use_hdim_cross_domain = False
                self.gdr = hdim_module
            else:
                self.gdr = GradeDecomposedRecurrence(cfg.grades)
        self.gdr_aux_proj = None
        if cfg.use_gdr and self.gdr is not None:
            self.gdr_aux_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        # NOTE: HRM is intentionally NOT wrapped in torch.compile here. Compiling
        # a submodule inside __init__ prefixes its state_dict keys with
        # `_orig_mod.` and breaks checkpoint compatibility. Use cfg.compile to
        # compile the whole model in the training entry point instead.
        self.hrm = (
            HRMCore(
                hidden_size=cfg.hidden_size,
                h_dim=cfg.h_dim,
                l_dim=cfg.l_dim,
                h_cycles=cfg.hrm_h_cycles,
                l_cycles=cfg.hrm_l_cycles,
            )
            if cfg.hrm
            else None
        )

        self.msa = None
        self.msa_router = None
        self.hdim_slot_router = None
        self.msa_registry = None
        if cfg.use_msa:
            self.msa = MSAAttention(
                hidden_size=cfg.hidden_size,
                num_query_heads=tcfg.num_query_heads,
                num_kv_heads=tcfg.num_kv_heads,
                rope_theta=tcfg.rope_theta,
                max_seq_len=tcfg.max_seq_len,
                use_binary_factorized=cfg.use_binary_factorized,
                binary_factorized_rank=cfg.binary_factorized_rank,
            )
            self.msa_router = SparseRouter(cfg.hidden_size, key_dim=64)
            self.hdim_slot_router = HDIMSlotRouter(cfg.hidden_size, key_dim=64)
            self.msa_registry = SlotRegistry(max_slots=cfg.msa_slot_count)

        self.nars_hrm = None
        self.nars_hdim = None
        self.nars_msa = None
        if cfg.use_nars:
            self.nars_hrm = NarsHrmController()
            self.nars_hdim = NarsHdimReasoner()
            self.nars_msa = NarsMsaReasoner()

        loops = cfg.loop_count if cfg.use_loop else 1
        self.iter_embed = nn.Parameter(torch.randn(loops, cfg.hidden_size) * 0.01)

        self.final_norm = RMSNorm(cfg.hidden_size, tcfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        self.quality_head = None
        if cfg.use_quality_head:
            self.quality_head = nn.Linear(cfg.hidden_size, 1, bias=True)

        # RoPE precomputed once up to max_seq_len (non-persistent buffers move
        # with .to(device); dtype cast is memoized in _rope_cache). The dict
        # cache below is only an overflow fallback for T > max_seq_len.
        head_dim = tcfg.hidden_size // tcfg.num_query_heads
        rope_cos, rope_sin = build_rope_cache(
            tcfg.max_seq_len,
            head_dim,
            tcfg.rope_theta,
            torch.device("cpu"),
            torch.float32,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self._rope = {}

        # Persisted step counter: the rotor schedule stays deterministic across
        # checkpoint save/resume. Old checkpoints without this key load fine
        # (see _load_from_state_dict).
        self._step = torch.zeros((), dtype=torch.long)
        _step = self._step
        del self._step
        self.register_buffer("_step", _step)

        self.apply(self._init_weights)

        # GPT-2 style depth-scaled init: residual-branch output projections are
        # scaled by 1/sqrt(2*L) so the residual stream variance stays bounded
        # with depth (and with recurrent reasoning loops).
        total_layers = (
            cfg.perception_layers + cfg.reasoning_layers + cfg.expression_layers
        )
        residual_scale = 1.0 / math.sqrt(2 * max(1, total_layers))
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                    p.mul_(residual_scale)
        # Re-assert weight tying (init must not silently untie).
        self.lm_head.weight = self.embed.weight

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Tolerant loading for checkpoints created before _step became a buffer.
        step_key = prefix + "_step"
        if step_key not in state_dict:
            state_dict[step_key] = self._step.detach().clone()
        elif not isinstance(state_dict[step_key], torch.Tensor):
            state_dict[step_key] = torch.tensor(
                int(state_dict[step_key]), dtype=torch.long
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _rope_cache(self, T: int, device, dtype, offset: int = 0):
        total = T + offset
        if total <= self.rope_cos.size(0):
            if self.rope_cos.device != device or self.rope_cos.dtype != dtype:
                self.rope_cos = self.rope_cos.to(device=device, dtype=dtype)
                self.rope_sin = self.rope_sin.to(device=device, dtype=dtype)
            return self.rope_cos[offset:total], self.rope_sin[offset:total]
        # Fallback for sequences beyond max_seq_len (kept small and bounded).
        key = (total, device, dtype)
        if key not in self._rope:
            head_dim = (
                self.cfg.transformer.hidden_size // self.cfg.transformer.num_query_heads
            )
            self._rope[key] = build_rope_cache(
                total, head_dim, self.cfg.transformer.rope_theta, device, dtype
            )
            if len(self._rope) > 8:
                self._rope.pop(next(iter(self._rope)))
        cos, sin = self._rope[key]
        return cos[offset:total], sin[offset:total]

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        ignore_index: int = -100,
        past_key_values=None,
        use_cache: bool = False,
        training_mode: bool = False,
        weights: dict[str, float] | None = None,
    ):
        """Returns logits, or (logits, loss) when targets are provided.

        nanoGPT-compatible. Targets are next-token labels aligned to input_ids
        (caller does the shift, or passes -100 for masked positions).

        When cfg.use_fused_ce is set and targets are provided, the loss is
        computed via the chunked fused lm_head+CE path and `logits` is None
        (full [B, T, V] logits are never materialized).
        """
        B, T = input_ids.shape
        cache_pos = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first = past_key_values[0]
            if first is not None:
                cache_pos = int(first[0].shape[2])
        h = self.embed(input_ids)
        cos, sin = self._rope_cache(T, h.device, h.dtype, cache_pos)
        next_key_values = [] if use_cache else None
        layer_idx = 0
        gdr_state = None
        pre_gdr_h = None
        use_gradient_checkpointing = (
            self.cfg.gradient_checkpointing and self.training and not use_cache
        )
        if self.training:
            self._step.add_(1)
        moe_aux_losses: list[torch.Tensor] = []
        collect_moe_aux = training_mode and (
            weights is None or weights.get("w_moe", 0.0) != 0.0
        )
        need_iso = training_mode and (
            weights is None or weights.get("w_iso", 0.0) != 0.0
        )
        need_quality = training_mode and (
            weights is None or weights.get("w_quality", 0.0) != 0.0
        )

        def run_block(block, hidden, past=None, gc: bool = True) -> Any:
            if use_gradient_checkpointing and gc:
                result = checkpoint(block, hidden, cos, sin, use_reentrant=False)
            elif use_cache:
                result = block(hidden, cos, sin, past, use_cache=True)
            else:
                result = block(hidden, cos, sin)
            if not use_cache and isinstance(result, tuple) and len(result) == 2:
                h_out, aux_loss = result
                if (
                    isinstance(aux_loss, torch.Tensor)
                    and aux_loss.ndim == 0
                    and collect_moe_aux
                ):
                    moe_aux_losses.append(aux_loss)
                return h_out
            return result

        def run_stage(blocks, hidden):
            """Run a sequence of transformer blocks, threading the KV cache."""
            nonlocal layer_idx
            for block in blocks:
                past = (
                    past_key_values[layer_idx]
                    if past_key_values is not None and layer_idx < len(past_key_values)
                    else None
                )
                if use_cache:
                    hidden, next_kv = run_block(block, hidden, past)  # type: ignore[assignment]
                    assert next_key_values is not None
                    next_key_values.append(next_kv)
                else:
                    hidden = run_block(block, hidden)  # type: ignore[assignment]
                layer_idx += 1
            return hidden

        h = run_stage(self.perception, h)

        # Precompute rotor index and gdr dispatch type once
        tgt_idx = None
        gdr_type = "none"
        if self.gdr is not None:
            if training_mode and hasattr(self.gdr, "rotors"):
                num_rotors = getattr(self.gdr.rotors, "num_rotors", 4)
                tgt_idx = _pick_rotor_idx(
                    self.cfg.rotor_seed, int(self._step.item()), num_rotors
                )
            if (
                training_mode
                and isinstance(self.gdr, DelayedHDIM)
                and self.gdr.delay_steps > 1
            ):
                gdr_type = "delayed"
            elif training_mode and isinstance(self.gdr, HDIMFull):
                gdr_type = "hdim"
            else:
                gdr_type = "default"

        if self.hrm is not None:
            if self.gdr is not None:
                if gdr_type == "delayed":
                    h, _, _, gdr_state, pre_gdr_h = self.hrm(
                        h,
                        self.reasoning,
                        cos,
                        sin,
                        gdr=self.gdr,
                        training_mode=training_mode,
                        gradient_checkpointing=use_gradient_checkpointing,
                        tgt_rotor_idx=tgt_idx,
                        moe_aux_losses=moe_aux_losses,
                        nars_controller=self.nars_hrm,
                    )
                elif gdr_type == "hdim":
                    gdr_state = self.gdr(
                        h, src_rotor_idx=0, tgt_rotor_idx=tgt_idx, return_state=True
                    )
                    pre_gdr_h = h if need_iso else None
                    h = gdr_state["fused"]
                    h, _, _, _, _ = self.hrm(
                        h,
                        self.reasoning,
                        cos,
                        sin,
                        training_mode=training_mode,
                        gradient_checkpointing=use_gradient_checkpointing,
                        moe_aux_losses=moe_aux_losses,
                        nars_controller=self.nars_hrm,
                    )
                else:
                    h = self.gdr(h)
                    h, _, _, _, _ = self.hrm(
                        h,
                        self.reasoning,
                        cos,
                        sin,
                        training_mode=training_mode,
                        gradient_checkpointing=use_gradient_checkpointing,
                        moe_aux_losses=moe_aux_losses,
                        nars_controller=self.nars_hrm,
                    )
            else:
                h, _, _, _, _ = self.hrm(
                    h,
                    self.reasoning,
                    cos,
                    sin,
                    training_mode=training_mode,
                    gradient_checkpointing=use_gradient_checkpointing,
                    moe_aux_losses=moe_aux_losses,
                    nars_controller=self.nars_hrm,
                )
            layer_idx += len(self.reasoning)
        else:
            loops = self.cfg.loop_count if self.cfg.use_loop else 1
            for i in range(loops):
                if self.gdr is not None and gdr_type == "delayed":
                    for j, block in enumerate(self.reasoning):
                        gdr_state = self.gdr(
                            h,
                            src_rotor_idx=0,
                            tgt_rotor_idx=tgt_idx,
                            return_state=True,
                            delay_step=i * len(self.reasoning) + j,
                        )
                        pre_gdr_h = h if need_iso else None
                        h = gdr_state["fused"]
                        h = run_stage((block,), h)
                else:
                    if self.gdr is not None:
                        if gdr_type == "hdim":
                            gdr_state = self.gdr(
                                h,
                                src_rotor_idx=0,
                                tgt_rotor_idx=tgt_idx,
                                return_state=True,
                            )
                            pre_gdr_h = h if need_iso else None
                            h = gdr_state["fused"]
                        else:
                            h = self.gdr(h)
                    h = run_stage(self.reasoning, h)
                if self.training and self.cfg.thinking_noise > 0.0:
                    h = h + torch.randn_like(h) * self.cfg.thinking_noise
                h = h + self.iter_embed[i]

        # MSA integration after reasoning / GDR
        msa_out = None
        msa_slot_ids = None
        msa_scores = None
        if self.cfg.use_msa and self.msa is not None:
            assert self.hdim_slot_router is not None
            assert self.msa_router is not None
            assert self.msa_registry is not None
            self.msa_registry.clear()
            b, t, _ = h.shape
            nkv = self.msa.num_kv_heads
            head_dim = self.msa.head_dim

            if hasattr(self.msa, "kv_proj"):
                kv = self.msa.kv_proj(h).view(b, t, 2 * nkv, head_dim).transpose(1, 2)
                k, v = kv.split(nkv, dim=1)
            else:
                k = self.msa.k_proj(h).view(b, t, nkv, head_dim).transpose(1, 2)
                v = self.msa.v_proj(h).view(b, t, nkv, head_dim).transpose(1, 2)

            slot_ids, routing_keys, k_caches, v_caches = (
                self.hdim_slot_router.batch_create_slots(
                    hidden_states=h,
                    k_cache=k,
                    v_cache=v,
                    slot_id_base=0,
                    domain_id=0,
                )
            )
            self.msa_registry.batch_register(slot_ids, routing_keys, k_caches, v_caches)

            nars_weights = None
            if self.cfg.use_nars and self.nars_msa is not None:
                with torch.no_grad():
                    query_nars = routing_keys.mean(dim=0)  # [key_dim]
                    top_k_ids, top_values = self.nars_msa.route_top_k_with_nars(
                        self.msa_registry, query_nars, self.cfg.msa_top_k
                    )
                    msa_slot_ids = top_k_ids.unsqueeze(0).unsqueeze(0).expand(b, t, -1)
                    msa_scores = top_values.unsqueeze(0).unsqueeze(0).expand(b, t, -1)
                    nars_weights = self.nars_msa.compute_attention_weights(msa_slot_ids)
            else:
                msa_slot_ids, _raw_scores, msa_weights = self.msa_router.route_top_k(
                    h, self.msa_registry, self.cfg.msa_top_k
                )
                msa_scores = msa_weights

            msa_out = self.msa(
                h, msa_slot_ids, self.msa_registry, nars_weights=nars_weights
            )
            h = h + msa_out

        h = run_stage(self.expression, h)

        pre_logits_hidden = (
            h if need_quality and self.quality_head is not None else None
        )
        h = self.final_norm(h)

        logits = None
        loss = None
        if targets is not None and self.cfg.use_fused_ce and not use_cache:
            # Memory-efficient opt-in path: lm_head + CE per chunk, the full
            # [B, T, V] logits tensor is never materialized.
            loss = fused_linear_cross_entropy(
                h,
                self.lm_head.weight,
                targets,
                ignore_index=ignore_index,
                chunk_size=(
                    self.cfg.ce_chunk_size if self.cfg.ce_chunk_size > 0 else 4096
                ),
            )
        else:
            logits = self.lm_head(h)
            if targets is not None:
                loss = cross_entropy_loss(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=ignore_index,
                    chunk_size=self.cfg.ce_chunk_size,
                )

        if (
            self.gdr_aux_proj is not None
            and gdr_state is not None
            and isinstance(gdr_state, dict)
        ):
            if "fused" in gdr_state:
                gdr_state["features"] = self.gdr_aux_proj(gdr_state["fused"])

        if training_mode:
            result = {"logits": logits}
            if loss is not None:
                result["loss"] = loss
            if moe_aux_losses:
                result["moe_aux_loss"] = torch.stack(moe_aux_losses).sum()
                result["num_moe_layers"] = len(moe_aux_losses)
            if (
                (weights is None or weights.get("w_aux", 0.0) != 0.0)
                and gdr_state is not None
                and isinstance(gdr_state, dict)
            ):
                if "fused" in gdr_state or "features" in gdr_state:
                    # inject batch-index labels for contrastive auxiliary loss
                    if "labels" not in gdr_state:
                        b, t, _ = h.shape
                        gdr_state["labels"] = (
                            torch.arange(b, device=h.device)
                            .unsqueeze(1)
                            .expand(b, t)
                            .reshape(-1)
                        )
                    if any(
                        k in gdr_state for k in ("features", "embeddings", "output")
                    ):
                        result["auxiliary_output"] = gdr_state
                    else:
                        result["auxiliary_output"] = {
                            "features": gdr_state["fused"],
                            "labels": gdr_state["labels"],
                        }
            if pre_gdr_h is not None and gdr_state is not None:
                # Use hidden states before/after HDIM for meaningful L_iso
                result["invariant_src"] = pre_gdr_h
                if "fused" in gdr_state and gdr_state["fused"] is not None:
                    result["invariant_tgt"] = gdr_state["fused"]
            if pre_logits_hidden is not None:
                result["model_output"] = pre_logits_hidden
            if msa_slot_ids is not None:
                result["msa_slot_ids"] = msa_slot_ids
                result["msa_scores"] = msa_scores
            if pre_logits_hidden is not None and self.quality_head is not None:
                result["quality_score"] = self.quality_head(pre_logits_hidden).squeeze(
                    -1
                )
            return result

        if loss is not None:
            return logits, loss
        if use_cache:
            return logits, next_key_values
        return logits

    def clear_rope_cache(self) -> None:
        """Clear the overflow RoPE cache (the precomputed buffers stay)."""
        self._rope.clear()

    def num_parameters(self, unique: bool = True) -> int:
        # Reasoning core params count once (shared) regardless of loop_count.
        return sum(p.numel() for p in self.parameters())
