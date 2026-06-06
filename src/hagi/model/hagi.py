"""HAGI model — Perception / Reasoning / Expression with optional GDR and MSA.

A single class covers all four ablation models via config flags:

    Model A (baseline): use_loop=False, use_gdr=False
    Model B (loop):     use_loop=True,  use_gdr=False
    Model C (HDIM):     use_loop=False, use_gdr=True   (Clifford bolted on, no loop)
    Model D (GDR):      use_loop=True,  use_gdr=True   (full HAGI)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..nars.adapters import NarsHdimReasoner, NarsHrmController, NarsMsaReasoner
from .gdr import GradeConfig, GradeDecomposedRecurrence
from .hdim_full import DelayedHDIM, HDIMFull
from .hrm_full import HRMCore
from .msa import HDIMSlotRouter, MSAAttention, SlotRegistry, SparseRouter
from .transformer import RMSNorm, TransformerBlock, TransformerConfig, build_rope_cache


def _pick_rotor_idx(seed: int, step: int, num_rotors: int) -> int:
    """Pick a target rotor index via CPU Python RNG (no GPU sync)."""
    if num_rotors <= 1:
        return 0
    return random.Random(int(seed) + int(step)).randint(1, num_rotors - 1)


@dataclass
class HAGIConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    perception_layers: int = 4
    reasoning_layers: int = 4
    expression_layers: int = 4
    loop_count: int = 1
    use_loop: bool = True
    use_gdr: bool = True
    hdim_full: bool = False
    hdim_heads: int = 4
    hdim_delay_steps: int = 1
    hrm: bool = False
    hrm_h_cycles: int = 1
    hrm_l_cycles: int = 2
    h_dim: int = 256
    l_dim: int = 256
    gradient_checkpointing: bool = False
    rotor_seed: int = 42
    use_hdim_cross_domain: bool = False
    use_msa: bool = False
    msa_slot_count: int = 100
    msa_top_k: int = 5
    use_nars: bool = False
    thinking_noise: float = 0.0
    use_quality_head: bool = False
    use_binary_factorized: bool = False
    binary_factorized_rank: int = 8
    use_moe: bool = False
    num_experts: int = 8
    moe_top_k: int = 2
    moe_intermediate_size: int | None = None
    moe_alpha: float = 0.01
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
            self.transformer.moe_intermediate_size = self.transformer.intermediate_size // self.num_experts
        if self.use_gdr and not self.hdim_full and not self.hrm:
            assert self.hidden_size == self.grades.hidden_size, (
                f"grade dims sum to {self.grades.hidden_size}, hidden is {self.hidden_size}"
            )


class HAGI(nn.Module):
    def __init__(self, cfg: HAGIConfig):
        super().__init__()
        self.cfg = cfg
        tcfg = cfg.transformer

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.perception = nn.ModuleList(TransformerBlock(tcfg) for _ in range(cfg.perception_layers))
        self.reasoning = nn.ModuleList(TransformerBlock(tcfg) for _ in range(cfg.reasoning_layers))
        self.expression = nn.ModuleList(TransformerBlock(tcfg) for _ in range(cfg.expression_layers))

        self.gdr = None
        if cfg.use_gdr:
            if cfg.hdim_full:
                hdim_module = DelayedHDIM(
                    hidden_size=cfg.hidden_size,
                    heads=cfg.hdim_heads,
                    delay_steps=cfg.hdim_delay_steps,
                ) if cfg.hdim_delay_steps > 1 else HDIMFull(hidden_size=cfg.hidden_size, heads=cfg.hdim_heads)
                if not getattr(cfg, "use_hdim_cross_domain", False):
                    hdim_module.use_hdim_cross_domain = False
                self.gdr = hdim_module
            else:
                self.gdr = GradeDecomposedRecurrence(cfg.grades)
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

        self._rope = {}
        self._step = 0
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def _rope_cache(self, T: int, device, dtype, offset: int = 0):
        if hasattr(self, "_rope_cos") and hasattr(self, "_rope_sin"):
            cos = self._rope_cos
            sin = self._rope_sin
            if cos.device != device or cos.dtype != dtype:
                cos = cos.to(device=device, dtype=dtype)
                sin = sin.to(device=device, dtype=dtype)
                self._rope_cos = cos
                self._rope_sin = sin
            assert isinstance(cos, torch.Tensor)
            assert isinstance(sin, torch.Tensor)
            return cos[offset : offset + T], sin[offset : offset + T]
        key = (T + offset, device, dtype)
        if key not in self._rope:
            head_dim = self.cfg.transformer.hidden_size // self.cfg.transformer.num_query_heads
            self._rope[key] = build_rope_cache(T + offset, head_dim, self.cfg.transformer.rope_theta, device, dtype)
            # Limit cache size to prevent unbounded growth
            if len(self._rope) > 100:
                oldest = next(iter(self._rope))
                del self._rope[oldest]
        cos, sin = self._rope[key]
        return cos[offset : offset + T], sin[offset : offset + T]

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        ignore_index: int = -100,
        past_key_values=None,
        use_cache: bool = False,
        training_mode: bool = False,
    ):
        """Returns logits, or (logits, loss) when targets are provided.

        nanoGPT-compatible. Targets are next-token labels aligned to input_ids
        (caller does the shift, or passes -100 for masked positions).
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
        gdr_output = None
        gdr_state = None
        pre_gdr_h = None
        use_gradient_checkpointing = self.cfg.gradient_checkpointing and self.training and not use_cache
        if self.training:
            self._step += 1
        moe_aux_losses: list[torch.Tensor] = []

        def run_block(block, hidden, past=None) -> Any:
            if use_gradient_checkpointing:
                result = checkpoint(
                    lambda h, c, s: block(h, c, s, gradient_checkpointing=True),
                    hidden,
                    cos,
                    sin,
                    use_reentrant=False,
                )
            elif use_cache:
                result = block(hidden, cos, sin, past, use_cache=True)
            else:
                result = block(hidden, cos, sin, gradient_checkpointing=self.cfg.gradient_checkpointing)
            if not use_cache and isinstance(result, tuple) and len(result) == 2:
                h_out, aux_loss = result
                if isinstance(aux_loss, torch.Tensor) and aux_loss.ndim == 0 and training_mode:
                    moe_aux_losses.append(aux_loss)
                return h_out
            return result

        for block in self.perception:
            past = past_key_values[layer_idx] if past_key_values is not None and layer_idx < len(past_key_values) else None
            if use_cache:
                h, next_kv = run_block(block, h, past)  # type: ignore[assignment]
                assert next_key_values is not None
                next_key_values.append(next_kv)
            else:
                h = run_block(block, h)  # type: ignore[assignment]
            layer_idx += 1

        if self.hrm is not None:
            if self.gdr is not None:
                assert self.gdr is not None
                if (
                    training_mode
                    and hasattr(self.gdr, "delay_steps")
                    and isinstance(self.gdr, (HDIMFull, DelayedHDIM))
                    and isinstance(self.gdr.delay_steps, int)
                    and self.gdr.delay_steps > 1
                ):
                    num_rotors = getattr(self.gdr.rotors, "num_rotors", 4)
                    tgt_idx = _pick_rotor_idx(self.cfg.rotor_seed, self._step, num_rotors)
                    h, _, _, gdr_state, pre_gdr_h = self.hrm(
                        h,
                        self.reasoning,
                        cos,
                        sin,
                        gdr=self.gdr,
                        training_mode=training_mode,
                        tgt_rotor_idx=tgt_idx,
                        moe_aux_losses=moe_aux_losses,
                        nars_controller=self.nars_hrm,
                    )
                elif training_mode and isinstance(self.gdr, HDIMFull):
                    num_rotors = getattr(self.gdr.rotors, "num_rotors", 4)
                    tgt_idx = _pick_rotor_idx(self.cfg.rotor_seed, self._step, num_rotors)
                    gdr_state = self.gdr(h, src_rotor_idx=0, tgt_rotor_idx=tgt_idx, return_state=True)
                    pre_gdr_h = h.clone()
                    h = gdr_state["fused"]
                    h, _, _, _, _ = self.hrm(h, self.reasoning, cos, sin, moe_aux_losses=moe_aux_losses, nars_controller=self.nars_hrm)
                else:
                    h = self.gdr(h)
                    h, _, _, _, _ = self.hrm(h, self.reasoning, cos, sin, moe_aux_losses=moe_aux_losses, nars_controller=self.nars_hrm)
                gdr_output = h
            else:
                h, _, _, _, _ = self.hrm(h, self.reasoning, cos, sin, moe_aux_losses=moe_aux_losses, nars_controller=self.nars_hrm)
            layer_idx += len(self.reasoning)
        else:
            loops = self.cfg.loop_count if self.cfg.use_loop else 1
            for i in range(loops):
                if self.gdr is not None:
                    assert self.gdr is not None
                    if (
                        training_mode
                        and hasattr(self.gdr, "delay_steps")
                        and isinstance(self.gdr, (HDIMFull, DelayedHDIM))
                        and isinstance(self.gdr.delay_steps, int)
                        and self.gdr.delay_steps > 1
                    ):
                        num_rotors = getattr(self.gdr.rotors, "num_rotors", 4)
                        tgt_idx = _pick_rotor_idx(self.cfg.rotor_seed, self._step, num_rotors)
                        for j, block in enumerate(self.reasoning):
                            current_step = i * len(self.reasoning) + j
                            gdr_state = self.gdr(
                                h,
                                src_rotor_idx=0,
                                tgt_rotor_idx=tgt_idx,
                                return_state=True,
                                delay_step=current_step,
                            )
                            pre_gdr_h = h.clone()
                            h = gdr_state["fused"]
                            gdr_output = h
                            past = past_key_values[layer_idx] if past_key_values is not None and layer_idx < len(past_key_values) else None
                            if use_cache:
                                h, next_kv = run_block(block, h, past)  # type: ignore[assignment]
                                assert next_key_values is not None
                                next_key_values.append(next_kv)
                            else:
                                h = run_block(block, h)  # type: ignore[assignment]
                            layer_idx += 1
                    elif training_mode and isinstance(self.gdr, HDIMFull):
                        num_rotors = getattr(self.gdr.rotors, "num_rotors", 4)
                        tgt_idx = _pick_rotor_idx(self.cfg.rotor_seed, self._step, num_rotors)
                        gdr_state = self.gdr(h, src_rotor_idx=0, tgt_rotor_idx=tgt_idx, return_state=True)
                        pre_gdr_h = h.clone()
                        h = gdr_state["fused"]
                        gdr_output = h
                        for block in self.reasoning:
                            past = past_key_values[layer_idx] if past_key_values is not None and layer_idx < len(past_key_values) else None
                            if use_cache:
                                h, next_kv = run_block(block, h, past)  # type: ignore[assignment]
                                assert next_key_values is not None
                                next_key_values.append(next_kv)
                            else:
                                h = run_block(block, h)  # type: ignore[assignment]
                            layer_idx += 1
                    else:
                        h = self.gdr(h)
                        gdr_output = h
                        for block in self.reasoning:
                            past = past_key_values[layer_idx] if past_key_values is not None and layer_idx < len(past_key_values) else None
                            if use_cache:
                                h, next_kv = run_block(block, h, past)  # type: ignore[assignment]
                                assert next_key_values is not None
                                next_key_values.append(next_kv)
                            else:
                                h = run_block(block, h)  # type: ignore[assignment]
                            layer_idx += 1
                else:
                    for block in self.reasoning:
                        past = past_key_values[layer_idx] if past_key_values is not None and layer_idx < len(past_key_values) else None
                        if use_cache:
                            h, next_kv = run_block(block, h, past)  # type: ignore[assignment]
                            assert next_key_values is not None
                            next_key_values.append(next_kv)
                        else:
                            h = run_block(block, h)  # type: ignore[assignment]
                        layer_idx += 1
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

            k = self.msa.k_proj(h).view(b, t, nkv, head_dim).transpose(1, 2)
            v = self.msa.v_proj(h).view(b, t, nkv, head_dim).transpose(1, 2)

            slots, routing_keys = self.hdim_slot_router.batch_create_slots(
                hidden_states=h,
                k_cache=k,
                v_cache=v,
                slot_id_base=0,
                domain_id=0,
            )
            self.msa_registry.batch_register(slots)
            self.msa_registry.set_routing_keys(routing_keys)

            nars_weights = None
            if self.cfg.use_nars and self.nars_msa is not None:
                with torch.no_grad():
                    inv = self.hdim_slot_router.routing_key(h)
                    query_nars = inv.mean(dim=(0, 1))  # [key_dim]
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

            msa_out = self.msa(h, msa_slot_ids, self.msa_registry, nars_weights=nars_weights)
            h = h + msa_out

        for block in self.expression:
            past = past_key_values[layer_idx] if past_key_values is not None and layer_idx < len(past_key_values) else None
            if use_cache:
                h, next_kv = run_block(block, h, past)  # type: ignore[assignment]
                assert next_key_values is not None
                next_key_values.append(next_kv)
            else:
                h = run_block(block, h)  # type: ignore[assignment]
            layer_idx += 1

        pre_logits_hidden = h.clone() if training_mode and self.quality_head is not None else None
        h = self.final_norm(h)
        logits = self.lm_head(h)

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        else:
            loss = None

        if training_mode:
            result = {"logits": logits}
            if loss is not None:
                result["loss"] = loss
            if moe_aux_losses:
                result["moe_aux_loss"] = sum(moe_aux_losses)
                result["num_moe_layers"] = len(moe_aux_losses)
            if gdr_state is not None and isinstance(gdr_state, dict):
                if "fused" in gdr_state or "features" in gdr_state:
                    # inject batch-index labels for contrastive auxiliary loss
                    if "labels" not in gdr_state:
                        b, t, _ = logits.shape
                        gdr_state["labels"] = torch.arange(b, device=logits.device).unsqueeze(1).expand(b, t).reshape(-1)
                    if any(k in gdr_state for k in ("features", "embeddings", "output")):
                        result["auxiliary_output"] = gdr_state
                    else:
                        result["auxiliary_output"] = {"features": gdr_state["fused"], "labels": gdr_state["labels"]}
            if gdr_state is not None:
                assert pre_gdr_h is not None
                if "invariant" in gdr_state and gdr_state["invariant"] is not None:
                    result["invariant_src"] = gdr_state["invariant"]
                    if "target_invariant" in gdr_state and gdr_state["target_invariant"] is not None:
                        result["invariant_tgt"] = gdr_state["target_invariant"]
            if pre_logits_hidden is not None:
                result["model_output"] = pre_logits_hidden
            if msa_slot_ids is not None:
                result["msa_slot_ids"] = msa_slot_ids
                result["msa_scores"] = msa_scores
            if self.quality_head is not None:
                result["quality_score"] = self.quality_head(pre_logits_hidden).squeeze(-1)
            return result

        if loss is not None:
            return logits, loss
        if use_cache:
            return logits, next_key_values
        return logits

    def clear_rope_cache(self) -> None:
        """Clear the RoPE cache to prevent memory growth during generation."""
        self._rope.clear()

    def num_parameters(self, unique: bool = True) -> int:
        # Reasoning core params count once (shared) regardless of loop_count.
        return sum(p.numel() for p in self.parameters())
