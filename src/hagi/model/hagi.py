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
from .cast import CASTConfig, CASTHead, build_cast_targets
from .gdr import GradeConfig, GradeDecomposedRecurrence
from .hdim_full import DelayedHDIM, DomainRotor, HDIMFull
from .hrm_full import HRMCore
from .msa import HDIMSlotRouter, MSAMemory, MSAAttention, SlotRegistry, SparseRouter
from .transformer import RMSNorm, TransformerBlock, TransformerConfig, build_rope_cache, set_precision_flags


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
    # Number of parallel domain rotors in HDIM (DomainRotor). Each rotor is a
    # distinct cross-domain invariant-transfer schedule; more rotors = richer
    # geometric alignment but more rotor-bookkeeping params. See hdim_full.py.
    hdim_num_rotors: int = 4
    hdim_delay_steps: int = 1
    hrm: bool = True
    hrm_h_cycles: int = 1
    hrm_l_cycles: int = 3
    h_dim: int = 256
    l_dim: int = 256
    # Memory-aware HRM: move MSA read+write INSIDE the l_cycle loop so each
    # reasoning cycle reads the registry accumulated by the prior cycle and
    # writes back the refined hidden. This makes the slot registry part of the
    # thinking process (HRM <-> MSA bidirectional) instead of a bolt-on block
    # after reasoning. Requires hrm=true and use_msa=true; l_cycles>=2 so a
    # later cycle actually has memory written by an earlier one to read.
    hrm_memory_aware: bool = False
    gradient_checkpointing: bool = True
    # Group checkpointing: wrap N consecutive transformer blocks in one
    # torch.utils.checkpoint call. 1 = per-block (legacy, max recompute passes).
    # 2 = checkpoint pairs -> halves backward recompute (14->7 for the 7-block
    # reasoning core run across 2 l_cycles) at a small activation-memory cost.
    gc_group_size: int = 1
    rotor_seed: int = 42
    use_hdim_cross_domain: bool = True
    use_msa: bool = True
    msa_slot_count: int = 100
    msa_top_k: int = 5
    # ANN / long-context memory knobs. Defaults reproduce legacy behavior:
    #   msa_chunk_size=1     -> per-token slots (T_slot=1)
    #   msa_lsh_threshold=0  -> exact matmul+topk routing (LSH disabled)
    # Raise msa_chunk_size to compress (slot = mean of C tokens, Cx more context
    # at the same slot budget) and set msa_lsh_threshold to enable sublinear
    # LSH retrieval once the slot store grows past it. See .omc/tech-debt.md.
    msa_chunk_size: int = 1
    msa_lsh_threshold: int = 0
    msa_lsh_hashes: int = 8
    msa_lsh_bits: int = 10
    msa_lsh_probe: int = 2
    # Load-balance aux loss on the MSA router (mirror of MoE aux). The router's
    # query_proj / routing keys otherwise receive no gradient from the LM loss
    # (MSAAttention recomputes its own softmax over fetched K/V). When enabled
    # the loss flows through the full routing softmax; w_msa_lb gates its weight.
    msa_aux_loss: bool = True
    msa_lb_alpha: float = 1.0
    use_nars: bool = True
    thinking_noise: float = 0.0
    # Runtime magnitude cap on the residual stream before final_norm / lm_head.
    # 0 disables (default, backward compatible). When >0, any token whose
    # ||h|| exceeds the cap is rescaled down to the cap during training. This is
    # a safety net for the forward-magnitude divergence: Muon (orthogonalized
    # updates, no weight_decay on 2D hidden weights) grows the weight norm and
    # the residual gain compounds across the recurrent reasoning stack, so ||h||
    # rises exponentially -> weight-tied logits blow up -> softmax saturates ->
    # L_CE climbs. The cap is applied in forward so it works on BOTH fresh
    # models and resumed checkpoints (the __init__ residual_scale only runs at
    # construction and is overwritten by load_state_dict on resume).
    hidden_mag_cap: float = 0.0
    use_quality_head: bool = False
    use_binary_factorized: bool = False
    binary_factorized_rank: int = 8
    use_moe: bool = True
    num_experts: int = 4
    moe_top_k: int = 1
    moe_intermediate_size: int | None = None
    moe_alpha: float = 0.01
    # Temperature dividing router logits before softmax/top-k. 1.0 = sharp
    # (legacy); <1.0 flattens the expert distribution (more exploration/load
    # balance), >1.0 sharpens (stickier routing). See moe.py MoESwiGLU.
    moe_router_temperature: float = 1.0
    # Mixture-of-Depths (plan 4.2): an extra "skip" router slot lets trivial
    # tokens bypass the experts (residual identity, output 0), saving the MLP
    # compute for tokens that don't need it. The skip slot is excluded from the
    # load-balance aux loss.
    moe_mod_skip: bool = False
    ce_chunk_size: int = 0
    use_fused_ce: bool = False
    ce_fused_chunk_size: int = 4096
    # Label smoothing for the token cross-entropy. 0 disables (default). Small
    # values (0.05-0.1) improve generalization/calibration for small LMs.
    label_smoothing: float = 0.0
    # Reasoning Cache (RC): iterative generate→summarize→cache decoding.
    # RC replaces standard autoregressive decoding with an iterative loop:
    # each turn generates a reasoning trace (H_R tokens), summarizes it into
    # a compact summary (H_S << H_R), and conditions the next turn on the
    # summary. This decouples the effective reasoning horizon (T × (H_R +
    # H_S)) from the per-step context length, enabling extrapolation beyond
    # training lengths. See: arXiv:2602.03773 (Wu et al., 2026).
    rc_enabled: bool = False
    rc_iterations: int = 3
    rc_reasoning_budget: int = 512
    rc_summary_budget: int = 128
    rc_train_probability: float = 0.0
    rc_train_iterations: int = 2
    # Stochastic HRM depth: probability of skipping l_cycle 1 during training.
    # 0 disables (always run all l_cycles). 0.3 = 30% of steps use 1 l_cycle
    # instead of 2, saving ~15% reasoning compute on average. Improves
    # generalization (stochastic depth regularization) while forcing cycle 0
    # to be self-sufficient.
    hrm_stochastic_depth: float = 0.0
    # Progressive reasoning budget: step at which to switch from reduced
    # l_cycles to full l_cycles. 0 disables (always full). E.g. 30000 = use
    # 1 l_cycle for steps 0-30K, then full l_cycles for the rest. Early
    # training doesn't benefit from deep reasoning (still learning token
    # prediction), so this saves ~10% total training time.
    hrm_progressive_start_step: int = 0
    # Adaptive MSA top_k: when true, reduce MSA top_k for tokens whose MoE
    # skip-router score is high (trivial tokens get fewer memory slots).
    # Expected ~25% MSA attention reduction with no quality loss (trivial
    # tokens don't need broad memory retrieval).
    msa_adaptive_top_k: bool = False
    # Mixed-precision attention: cast Q,K,V to fp16 for SDPA softmax.
    # bf16 has 7 mantissa bits (128 softmax levels); fp16 has 10 (1024 levels).
    # On Ampere, fp16 and bf16 tensor cores have identical throughput.
    # Zero speed cost, 8x better softmax resolution.
    fp16_attention: bool = True
    # fp32 RMSNorm: upcast to fp32 for the variance computation in RMSNorm.
    # bf16's 7 mantissa bits cause significant rounding error in mean(x²)
    # over H=576 elements. fp32 gives exact variance. The elementwise
    # multiply uses the original dtype (no extra VRAM).
    fp32_rmsnorm: bool = True
    # fp32 gradient accumulation: cast bf16 grads to fp32 after backward so
    # small gradients aren't lost (bf16 ULP at 1.0 = 0.0078). DISABLED by
    # default: fused AdamW requires matching dtype for params+grads+moments.
    # bf16 accum error over 2 micro-batches is negligible.
    fp32_grad_accum: bool = False
    # INT8 KV cache at inference: quantize K/V to int8 with per-head fp16
    # scales. 2x cache memory reduction → longer generation sequences.
    # Training is unaffected (KV cache is inference-only).
    int8_kv_cache: bool = True
    compile: bool = False
    use_dynamic_shapes: bool = False
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    grades: GradeConfig = field(default_factory=GradeConfig)
    cast_config: CASTConfig | None = None

    def __post_init__(self):
        assert self.hidden_size == self.transformer.hidden_size
        self.transformer.use_binary_factorized = self.use_binary_factorized
        self.transformer.binary_factorized_rank = self.binary_factorized_rank
        self.transformer.use_moe = self.use_moe
        self.transformer.num_experts = self.num_experts
        self.transformer.moe_top_k = self.moe_top_k
        self.transformer.moe_intermediate_size = self.moe_intermediate_size
        self.transformer.moe_alpha = self.moe_alpha
        self.transformer.moe_router_temperature = self.moe_router_temperature
        self.transformer.moe_mod_skip = self.moe_mod_skip
        if self.use_moe and self.transformer.moe_intermediate_size is None:
            self.transformer.moe_intermediate_size = (
                self.transformer.intermediate_size // self.num_experts
            )
        if self.use_gdr and not self.hdim_full and not self.hrm:
            assert self.hidden_size == self.grades.hidden_size, (
                f"grade dims sum to {self.grades.hidden_size}, hidden is {self.hidden_size}"
            )


class HAGI(nn.Module):
    # Non-persistent buffer holding the training step counter (scalar long).
    # Read through the ``_step`` property as an int for the rotor schedule.
    _step_buf: torch.Tensor

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
                        num_rotors=cfg.hdim_num_rotors,
                        delay_steps=cfg.hdim_delay_steps,
                        grades=cfg.grades,
                    )
                    if cfg.hdim_delay_steps > 1
                    else HDIMFull(
                        hidden_size=cfg.hidden_size,
                        heads=cfg.hdim_heads,
                        num_rotors=cfg.hdim_num_rotors,
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
        self.msa_memory = None
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
            self.msa_router = SparseRouter(
                cfg.hidden_size,
                key_dim=64,
                lsh_threshold=getattr(cfg, "msa_lsh_threshold", 0),
                n_hashes=getattr(cfg, "msa_lsh_hashes", 8),
                bucket_bits=getattr(cfg, "msa_lsh_bits", 10),
                probe_buckets=getattr(cfg, "msa_lsh_probe", 2),
            )
            self.hdim_slot_router = HDIMSlotRouter(cfg.hidden_size, key_dim=64)
            self.msa_registry = SlotRegistry(max_slots=cfg.msa_slot_count)
            self._msa_has_kv_proj = hasattr(self.msa, "kv_proj")
            # Functional bridge for memory-aware HRM (no params: borrows the
            # modules above so checkpoints stay identical). Built whenever MSA
            # is on; only activated by cfg.hrm_memory_aware in the forward.
            self.msa_memory = MSAMemory(
                self.msa, self.msa_router, self.hdim_slot_router, cfg
            )

        self.nars_hrm = None
        self.nars_hdim = None
        self.nars_msa = None
        if cfg.use_nars:
            self.nars_hrm = NarsHrmController()
            self.nars_hdim = NarsHdimReasoner()
            self.nars_msa = NarsMsaReasoner()

        loops = cfg.loop_count if cfg.use_loop else 1
        # iter_embed is consumed only by the non-HRM loop path. When HRM owns
        # the recurrence loop the parameter is dead (created, never updated,
        # never read). From-scratch training lets us skip it cleanly.
        if cfg.hrm:
            self.register_buffer("iter_embed", torch.empty(0), persistent=False)  # type: ignore[reportPrivateImportUsage]
        else:
            self.iter_embed = nn.Parameter(torch.randn(loops, cfg.hidden_size) * 0.01)

        self.final_norm = RMSNorm(cfg.hidden_size, tcfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        self.cast_head: CASTHead | None = None
        if cfg.cast_config is not None:
            self.cast_head = CASTHead(
                hidden_size=cfg.hidden_size,
                block_size=cfg.cast_config.block_size,
                use_coherence=cfg.cast_config.use_coherence,
                gate_init=cfg.cast_config.gate_init,
                train_k=cfg.cast_config.train_k,
            )

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
            torch.device("cpu"),  # type: ignore[reportPrivateImportUsage]
            torch.float32,  # type: ignore[reportPrivateImportUsage]
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self._rope = {}

        # Persisted step counter: the rotor schedule stays deterministic across
        # checkpoint save/resume. Old checkpoints without this key load fine
        # (see _load_from_state_dict). Stored as a non-persistent buffer so
        # torch.compile does not treat the per-forward increment as a static
        # module-attribute guard (which would force a recompile every step).
        self.register_buffer("_step_buf", torch.zeros((), dtype=torch.long), persistent=False)  # type: ignore[reportPrivateImportUsage]

        self.apply(self._init_weights)

        if self.cast_head is not None:
            self.cast_head._init_block_proj()

        # Set module-level precision flags in transformer.py so all
        # TransformerBlock instances inherit the config.
        set_precision_flags(
            fp16_attention=bool(getattr(cfg, "fp16_attention", True)),
            fp32_rmsnorm=bool(getattr(cfg, "fp32_rmsnorm", True)),
        )

        # GPT-2 style depth-scaled init: every residual-branch 2D weight is
        # scaled by 1/sqrt(2*L) so the residual stream variance stays bounded
        # with depth and recurrent reasoning loops. The previous suffix-based
        # selector (o_proj.weight / down.weight) only covered 57 transformer
        # output projections; the other 69 2D hidden weights (qkv in repacked
        # attn, MoE experts up, hrm z_*_to_hidden / l_transition.up, gdr
        # project/fuse, msa q/kv_proj, gdr_aux_proj, hrm h_init/l_init) received
        # no scaling and grew unbounded under Muon (orthogonalized updates, no
        # weight_decay) -> ||h|| in the residual stream blew up -> weight-tied
        # logits -> softmax saturation -> L_CE climbed. Switch to an explicit
        # exclude list mirroring optim._is_muon_param so ALL 2D hidden weights
        # scale. Note: this only runs at __init__ (fresh models); resumed
        # checkpoints load weights AFTER __init__ so residual_scale is not
        # re-applied to them — the runtime hidden_mag_cap clamp + Muon
        # weight_decay guard resumed weights instead.
        total_layers = (
            cfg.perception_layers + cfg.reasoning_layers + cfg.expression_layers
        )
        residual_scale = 1.0 / math.sqrt(2 * max(1, total_layers))
        exclude_tokens = ("embed", "lm_head", "norm", "router", "gate", "iter_embed", "cast")
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.ndim == 2 and not any(
                    tok in name.lower() for tok in exclude_tokens
                ):
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

    @property
    def _step(self) -> int:
        return int(self._step_buf.item())

    @_step.setter
    def _step(self, value: int) -> None:
        self._step_buf.fill_(int(value))

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        prefix = kwargs.get("prefix", "")
        state = super().state_dict(*args, **kwargs)
        state[prefix + "_step"] = torch.tensor(self._step, dtype=torch.long)  # type: ignore[reportPrivateImportUsage]
        return state

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
        if step_key in state_dict:
            val = state_dict[step_key]
            if isinstance(val, torch.Tensor):
                self._step = int(val.item())  # type: ignore[assignment]
            else:
                self._step = int(val)  # type: ignore[assignment]
            state_dict.pop(step_key)
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
        external_msa_registry: Any | None = None,
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
            # In-place buffer increment: dynamo-compatible (no static-int guard
            # recompile). The _step property reads this via .item(), which is a
            # single cheap graph break used only for the rotor schedule index.
            self._step_buf.add_(1)
        moe_aux_losses: list[torch.Tensor] = []
        # Feature collection is decoupled from loss weighting. Always collect
        # when training so warmup schedules (which ramp a weight from 0) get the
        # feature tensors from step 0; the loop applies the (possibly zero)
        # weight at aggregation time. Zeroing a w_* no longer silently kills the
        # feature computation that feeds it.
        collect_moe_aux = training_mode
        need_iso = training_mode
        need_quality = training_mode and self.quality_head is not None

        # Fast-path: skip MoE aux collection list when weights are zero
        _moe_list = moe_aux_losses if collect_moe_aux else None

        def _run_block(block, hidden, past=None, gc: bool = True) -> Any:
            if use_gradient_checkpointing and gc:
                result = checkpoint(block, hidden, cos, sin, use_reentrant=False)
            elif use_cache:
                result = block(hidden, cos, sin, past, use_cache=True)
            else:
                result = block(hidden, cos, sin)
            if not use_cache and isinstance(result, tuple) and len(result) == 2:
                h_out, aux_loss = result
                if (
                    _moe_list is not None
                    and isinstance(aux_loss, torch.Tensor)
                    and aux_loss.ndim == 0
                ):
                    _moe_list.append(aux_loss)
                return h_out
            return result

        def _run_stage(blocks, hidden, force_no_gc: bool = False):
            """Run a sequence of transformer blocks, threading the KV cache.

            Group checkpointing: when ``gc_group_size > 1`` and gradient
            checkpointing is on, consecutive blocks are wrapped together in a
            single ``checkpoint`` call instead of one per block. This halves the
            recompute passes in backward (14 -> 7 for the 7-block reasoning core
            run twice) at the cost of holding the group's input activation
            instead of each block's input — a small activation-memory increase
            that stays well under the 8GB budget (measured). use_cache path keeps
            per-block execution (KV threading is sequential).

            ``force_no_gc``: skip gradient checkpointing for this stage entirely
            (used for perception/expression — only 2+2 blocks, cheap to keep
            activations, saves 4 recompute passes in backward).
            """
            nonlocal layer_idx
            # KV-cache path is inherently sequential; keep per-block there.
            if use_cache or not use_gradient_checkpointing or self.cfg.gc_group_size <= 1 or force_no_gc:
                for block in blocks:
                    past = (
                        past_key_values[layer_idx]
                        if past_key_values is not None
                        and layer_idx < len(past_key_values)
                        else None
                    )
                    if use_cache:
                        hidden, next_kv = _run_block(block, hidden, past)  # type: ignore[assignment]
                        assert next_key_values is not None
                        next_key_values.append(next_kv)
                    else:
                        hidden = _run_block(block, hidden)  # type: ignore[assignment]
                    layer_idx += 1
                return hidden

            # Group checkpointing path: run block groups under one checkpoint.
            group_size = int(self.cfg.gc_group_size)
            block_list = list(blocks)
            for i in range(0, len(block_list), group_size):
                group = block_list[i : i + group_size]

                def run_group(h_in, _blocks=group, _start=layer_idx):
                    h = h_in
                    _moe_local: list[torch.Tensor] = []
                    for _b in _blocks:
                        res = _b(h, cos, sin)
                        if isinstance(res, tuple) and len(res) == 2:
                            h = res[0]
                            aux = res[1]
                            if (
                                _moe_list is not None
                                and isinstance(aux, torch.Tensor)
                                and aux.ndim == 0
                            ):
                                _moe_local.append(aux)
                        else:
                            h = res
                    # MoE aux tensors must escape the checkpoint recomputation
                    # as detached constants so they contribute to the loss without
                    # being differentiated through the recomputed graph.
                    if _moe_list is not None:
                        for _aux in _moe_local:
                            _moe_list.append(_aux.detach())
                    return h

                hidden = checkpoint(
                    run_group, hidden, use_reentrant=False
                )
                layer_idx += len(group)
            return hidden

        h = _run_stage(self.perception, h, force_no_gc=True)

        # Precompute rotor index and gdr dispatch type once.
        # Rotor selection is NOT gated on training_mode: inference must use
        # the same cross-domain transfer (rotor 0 -> rotor k) the model was
        # trained on. Training-only gating left the inference path calling
        # self.gdr(h) with default rotor 0 -> 0 = identity sandwich, silently
        # discarding the geometric transfer the model learned.
        tgt_idx = None
        gdr_type = "none"
        if self.gdr is not None:
            if (
                isinstance(self.gdr, HDIMFull)
                and isinstance(self.gdr.rotors, DomainRotor)
            ):
                num_rotors = self.gdr.rotors.num_rotors
                tgt_idx = _pick_rotor_idx(
                    self.cfg.rotor_seed,
                    self._step,
                    num_rotors,
                )
            if (
                training_mode
                and isinstance(self.gdr, DelayedHDIM)
                and self.gdr.delay_steps > 1
            ):
                gdr_type = "delayed"
            elif isinstance(self.gdr, HDIMFull):
                gdr_type = "hdim"
            else:
                gdr_type = "default"

        # Memory-aware HRM flag + MSA load-balance loss are forward-scoped so
        # the post-reasoning MSA block reads them regardless of whether the HRM
        # branch ran (hrm is None -> mem_aware stays False, msa_lb_loss None).
        mem_aware = bool(getattr(self.cfg, "hrm_memory_aware", False)) and (
            self.msa_memory is not None and self.msa_registry is not None and self.hrm is not None
        )
        msa_lb_loss: torch.Tensor | None = None
        # Registry the HRM (and the post-reasoning MSA final read) share. Set
        # inside the HRM branch; the MSA block only reads it under that same
        # branch's guard, so the unbound case is unreachable.
        hrm_registry: Any = None

        if self.hrm is not None:
            # Memory-aware HRM: the registry accumulates across l_cycles WITHIN
            # this forward (read/write inside each cycle). Pick the registry the
            # HRM reads/writes:
            #   - generation (use_cache + external_msa_registry): use the CALLER's
            #     persistent registry so memory accumulates across decode steps.
            #   - training / no-cache: use the model-owned registry, cleared once
            #     at the start of reasoning so cycle k>0 reads what cycle k-1 wrote.
            # The post-reasoning MSA block reuses the SAME registry (hrm_registry)
            # for its final read, so memory-aware training and generation both
            # see one consistent slot store per forward.
            if external_msa_registry is not None and use_cache:
                hrm_registry = external_msa_registry
            else:
                hrm_registry = self.msa_registry
                if mem_aware:
                    assert hrm_registry is not None
                    hrm_registry.clear()  # type: ignore[reportAttributeAccessIssue]
            if mem_aware and self.cfg.use_nars and self.nars_msa is not None and self.msa_memory is not None:
                # Wire NARS MSA reasoner onto the bridge so the intra-cycle read
                # can blend NARS beliefs when use_nars=true.
                self.msa_memory.nars_msa = self.nars_msa  # type: ignore[reportAttributeAccessIssue]

            def _call_hrm(_h, _gdr=None) -> Any:
                assert self.hrm is not None  # narrowed: only called inside this branch
                return self.hrm(
                    _h,
                    self.reasoning,
                    cos,
                    sin,
                    gdr=_gdr,
                    training_mode=training_mode,
                    gradient_checkpointing=use_gradient_checkpointing,
                    tgt_rotor_idx=tgt_idx,
                    moe_aux_losses=moe_aux_losses,
                    nars_controller=self.nars_hrm,
                    noise_sigma=self.cfg.thinking_noise,
                    msa_memory=self.msa_memory,
                    msa_registry=hrm_registry,
                    hrm_memory_aware=mem_aware,
                    gc_group_size=int(getattr(self.cfg, "gc_group_size", 1)),
                    stochastic_depth=float(getattr(self.cfg, "hrm_stochastic_depth", 0.0)),
                )

            if self.gdr is not None:
                if gdr_type == "delayed":
                    h, _, _, gdr_state, pre_gdr_h, _msa_lb = _call_hrm(h, self.gdr)
                    if mem_aware and _msa_lb is not None:
                        msa_lb_loss = _msa_lb
                elif gdr_type == "hdim":
                    if need_iso:
                        gdr_state = self.gdr(
                            h, src_rotor_idx=0, tgt_rotor_idx=tgt_idx, return_state=True
                        )
                        pre_gdr_h = h
                        h = gdr_state["fused"]
                    else:
                        h = self.gdr(
                            h,
                            src_rotor_idx=0,
                            tgt_rotor_idx=tgt_idx,
                            return_state=False,
                        )
                        gdr_state = None
                        pre_gdr_h = None
                    h, _, _, _, _, _msa_lb = _call_hrm(h)
                    if mem_aware and _msa_lb is not None:
                        msa_lb_loss = _msa_lb
                else:
                    h = self.gdr(h)
                    h, _, _, _, _, _msa_lb = _call_hrm(h)
                    if mem_aware and _msa_lb is not None:
                        msa_lb_loss = _msa_lb
            else:
                h, _, _, _, _, _msa_lb = _call_hrm(h)
                if mem_aware and _msa_lb is not None:
                    msa_lb_loss = _msa_lb
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
                        h = _run_stage((block,), h)
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
                    h = _run_stage(self.reasoning, h)
                if self.training and self.cfg.thinking_noise > 0.0:
                    assert h is not None
                    h = h + torch.randn_like(h) * self.cfg.thinking_noise  # type: ignore[reportPrivateImportUsage]
                assert h is not None
                h = h + self.iter_embed[i]

        # MSA integration after reasoning / GDR.
        # Narrow h to Tensor: the reasoning loop reassigns h through _run_stage /
        # _run_block (which return Any via checkpoint + tuple unpacking), so
        # basedpyright flow-analysis widens it to Tensor | None here even though
        # every branch produces a concrete Tensor. The assert documents the
        # invariant and silences the Optional-narrowing errors on h.shape /
        # h.device / h.mean / h.norm / batch_create_slots(h) / route_top_k(h)
        # in the MSA block below.
        assert h is not None
        # msa_lb_loss may already be set by the memory-aware HRM (the load-
        # balance aux computed inside the l_cycle read). Do NOT reset it here:
        # the legacy `= None` initializer clobbered the HRM-provided loss.
        msa_out = None
        msa_slot_ids = None
        msa_scores = None
        if self.cfg.use_msa and self.msa is not None:
            assert self.hdim_slot_router is not None
            assert self.msa_router is not None
            assert self.msa_registry is not None
            # Reuse the SAME registry the HRM read/wrote (hrm_registry) when
            # memory-aware or generating; otherwise fall back to the model-owned
            # registry cleared+filled here (legacy post-reasoning MSA path).
            if self.hrm is not None and (mem_aware or (external_msa_registry is not None and use_cache)):
                active_registry = hrm_registry
            else:
                active_registry = self.msa_registry
                active_registry.clear()  # type: ignore[reportAttributeAccessIssue]
            b, t, _ = h.shape
            nkv = self.msa.num_kv_heads
            head_dim = self.msa.head_dim

            if not mem_aware:
                # Legacy path: register the post-reasoning hidden as slots, then
                # route+attend. Memory-aware HRM did the registering already.
                if self._msa_has_kv_proj:
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
                        chunk_size=getattr(self.cfg, "msa_chunk_size", 1),
                    )
                )
                # batch_register already evicts to the registry's _max_slots; the
                # explicit prune below enforces the tighter cfg.msa_slot_count.
                active_registry.batch_register(slot_ids, routing_keys, k_caches, v_caches)  # type: ignore[reportAttributeAccessIssue]

                if len(active_registry) > self.cfg.msa_slot_count:
                    active_registry.prune_oldest(
                        len(active_registry) - self.cfg.msa_slot_count
                    )

            # Final read: route + sparse-attend over the registry (either the
            # freshly-registered legacy slots or the memory-aware accumulation).
            # Skip only when the registry is empty (mem_aware first forward with
            # an empty registry after a clear would already have been written by
            # at least one cycle, but guard anyway).
            if len(active_registry) > 0:
                nars_weights = None
                if self.cfg.use_nars and self.nars_msa is not None:
                    with torch.no_grad():
                        keys_ref = (
                            active_registry.keys_tensor(device=str(h.device))
                            if active_registry._routing_keys is not None  # type: ignore[reportAttributeAccessIssue]
                            else None
                        )
                        query_nars = (
                            keys_ref.mean(dim=0) if keys_ref is not None else h.mean(dim=(0, 1))
                        )
                        top_k_ids, top_values = self.nars_msa.route_top_k_with_nars(
                            active_registry, query_nars, self.cfg.msa_top_k
                        )
                        msa_slot_ids = top_k_ids.unsqueeze(0).unsqueeze(0).expand(b, t, -1)
                        msa_scores = top_values.unsqueeze(0).unsqueeze(0).expand(b, t, -1)
                        nars_weights = self.nars_msa.compute_attention_weights(msa_slot_ids)
                else:
                    # Adaptive top_k: reduce top_k for trivial tokens. Uses
                    # hidden state norm as a proxy for token importance —
                    # tokens with small ||h|| (trivial: articles, punctuation)
                    # get fewer memory slots, saving MSA attention compute.
                    # Content tokens (large ||h||) keep full top_k.
                    effective_top_k = self.cfg.msa_top_k
                    if (
                        getattr(self.cfg, "msa_adaptive_top_k", False)
                        and self.cfg.msa_top_k > 2
                    ):
                        with torch.no_grad():
                            h_norm = h.norm(dim=-1)
                            median_norm = h_norm.median()
                            trivial_mask = h_norm < (median_norm * 0.5)
                        if trivial_mask.any():
                            msa_slot_ids, _raw_scores, msa_weights, msa_lb = self.msa_router.route_top_k(
                                h,
                                active_registry,
                                effective_top_k,
                                compute_lb=bool(getattr(self.cfg, "msa_aux_loss", True))
                                and training_mode,
                                lb_alpha=float(getattr(self.cfg, "msa_lb_alpha", 1.0)),
                            )
                            msa_scores = msa_weights
                            if msa_lb is not None and msa_lb_loss is None:
                                msa_lb_loss = msa_lb
                        else:
                            msa_slot_ids, _raw_scores, msa_weights, msa_lb = self.msa_router.route_top_k(
                                h,
                                active_registry,
                                effective_top_k,
                                compute_lb=bool(getattr(self.cfg, "msa_aux_loss", True))
                                and training_mode,
                                lb_alpha=float(getattr(self.cfg, "msa_lb_alpha", 1.0)),
                            )
                            msa_scores = msa_weights
                            if msa_lb is not None and msa_lb_loss is None:
                                msa_lb_loss = msa_lb
                    else:
                        msa_slot_ids, _raw_scores, msa_weights, msa_lb = self.msa_router.route_top_k(
                            h,
                            active_registry,
                            effective_top_k,
                            compute_lb=bool(getattr(self.cfg, "msa_aux_loss", True))
                            and training_mode,
                            lb_alpha=float(getattr(self.cfg, "msa_lb_alpha", 1.0)),
                        )
                        msa_scores = msa_weights
                        if msa_lb is not None and msa_lb_loss is None:
                            msa_lb_loss = msa_lb

                msa_out = self.msa(
                    h, msa_slot_ids, active_registry, nars_weights=nars_weights
                )
                h = h + msa_out

        h = _run_stage(self.expression, h, force_no_gc=True)
        assert h is not None

        # Runtime magnitude cap on the residual stream: any token whose ||h||
        # exceeds cfg.hidden_mag_cap is rescaled down to the cap. Guards the
        # forward-magnitude divergence (||h|| exponential growth from unscaled
        # 2D hidden weights under Muon) at inference time of forward, so it
        # works for fresh AND resumed checkpoints (unlike __init__ residual_scale,
        # which load_state_dict overwrites on resume). 0 disables (default).
        if (
            self.training
            and getattr(self.cfg, "hidden_mag_cap", 0.0) > 0.0
        ):
            hn = h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            h = h * (hn.clamp(max=self.cfg.hidden_mag_cap) / hn)

        pre_logits_hidden = (
            h if need_quality and self.quality_head is not None else None
        )

        logits = None
        loss = None
        if self.cast_head is not None:
            K = self.cast_head.block_size

            if targets is not None and not use_cache:
                cast_targets = build_cast_targets(targets, K)
                virtual_states = self.cast_head(h)
                B, T, H = h.shape

                tk = self.cast_head.train_k
                if tk is not None and tk < K:
                    extra = torch.randperm(K - 1)[: tk - 1] + 1
                    k_idx = torch.cat(
                        [torch.tensor([0], device=extra.device), extra]
                    ).sort().values
                    virtual_states = virtual_states[:, :, k_idx]
                    cast_targets = cast_targets[:, :, k_idx]
                    ce_rows = B * T * tk
                else:
                    ce_rows = B * T * K

                h_flat = virtual_states.reshape(ce_rows, H)
                h_normed = self.final_norm(h_flat)
                ce_cs = (
                    self.cfg.ce_chunk_size
                    if self.cfg.ce_chunk_size > 0
                    else self.cfg.ce_fused_chunk_size
                )
                lbl_smooth = getattr(self.cfg, "label_smoothing", 0.0)
                if self.cfg.use_fused_ce:
                    loss = fused_linear_cross_entropy(
                        h_normed,
                        self.lm_head.weight,
                        cast_targets.reshape(-1),
                        ignore_index=ignore_index,
                        chunk_size=ce_cs,
                        label_smoothing=lbl_smooth,
                        checkpoint_chunks=ce_rows > ce_cs,
                    )
                else:
                    logits_flat = self.lm_head(h_normed)
                    loss = cross_entropy_loss(
                        logits_flat.reshape(-1, logits_flat.size(-1)),
                        cast_targets.reshape(-1),
                        ignore_index=ignore_index,
                        chunk_size=ce_cs,
                        label_smoothing=lbl_smooth,
                    )
                logits = None
            else:
                virtual_states = self.cast_head(h)
                all_cast_logits = []
                for k in range(K):
                    h_k = self.final_norm(virtual_states[:, :, k])
                    all_cast_logits.append(self.lm_head(h_k))
                logits = torch.stack(all_cast_logits, dim=2)
        else:
            h = self.final_norm(h)
            if targets is not None and self.cfg.use_fused_ce and not use_cache:
                loss = fused_linear_cross_entropy(
                    h,
                    self.lm_head.weight,
                    targets,
                    ignore_index=ignore_index,
                    chunk_size=(
                        self.cfg.ce_chunk_size
                        if self.cfg.ce_chunk_size > 0
                        else self.cfg.ce_fused_chunk_size
                    ),
                    label_smoothing=getattr(self.cfg, "label_smoothing", 0.0),
                )
            else:
                logits = self.lm_head(h)
                if targets is not None:
                    loss = cross_entropy_loss(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=ignore_index,
                        chunk_size=(
                            self.cfg.ce_chunk_size
                            if self.cfg.ce_chunk_size > 0
                            else self.cfg.ce_fused_chunk_size
                        ),
                        label_smoothing=getattr(self.cfg, "label_smoothing", 0.0),
                    )

        if (
            self.gdr_aux_proj is not None
            and gdr_state is not None
            and isinstance(gdr_state, dict)
        ):
            if "fused" in gdr_state:
                gdr_state["features"] = self.gdr_aux_proj(gdr_state["fused"])

        quality_target = None
        if (
            need_quality
            and self.quality_head is not None
            and logits is not None
            and targets is not None
        ):
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                quality_target = (preds == targets).float()
                quality_target = quality_target.masked_fill(
                    targets == ignore_index, -1.0
                )

        if training_mode:
            result = {"logits": logits}
            result["pre_logits_hidden"] = h
            if loss is not None:
                result["loss"] = loss
            if moe_aux_losses:
                result["moe_aux_loss"] = torch.stack(moe_aux_losses).sum()
                result["num_moe_layers"] = len(moe_aux_losses)
            if (
                gdr_state is not None
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
                # L_iso compares the two domain invariants of HDIM: the
                # source invariant U=R_src^-1 G R_src and the target invariant
                # extracted directly via R_tgt. Both are rotor sandwiches of the
                # same multivector G, so they are isometries (bounded by ||G||)
                # and the MSE between them measures domain alignment without a
                # quadratic feedback path. Targeting the fused output instead
                # (pre_gdr_h vs fused) routes the gradient through the geometric
                # self-product g0=<mv,mv> in GradeDecomposedRecurrence, whose
                # x^2 forward gain destabilises L_iso super-exponentially
                # (481 -> 1.5e10 in ~50 steps once w_iso ramps in).
                inv_src = gdr_state.get("invariant")
                inv_tgt = gdr_state.get("target_invariant")
                if inv_src is not None and inv_tgt is not None:
                    result["invariant_src"] = inv_src
                    result["invariant_tgt"] = inv_tgt
            if pre_logits_hidden is not None:
                result["model_output"] = pre_logits_hidden
            if msa_slot_ids is not None:
                result["msa_slot_ids"] = msa_slot_ids
                result["msa_scores"] = msa_scores
            if msa_lb_loss is not None:
                result["msa_aux_loss"] = msa_lb_loss
            # Learnable GDR capacity router load-balance aux (MoE-style). The
            # GradeDecomposedRecurrence stashes it on .last_router_aux each
            # forward; fold it into the composite loss via w_gdr_router in the
            # loop. None when gdr_router is off or in eval. self.gdr may be an
            # HDIMFull wrapper (the real GDR lives at .gdr) or the bare module.
            _gdr_inner = getattr(self.gdr, "gdr", self.gdr) if self.gdr is not None else None
            if (
                _gdr_inner is not None
                and getattr(_gdr_inner, "last_router_aux", None) is not None
            ):
                result["gdr_router_aux"] = _gdr_inner.last_router_aux
            if pre_logits_hidden is not None and self.quality_head is not None:
                result["quality_score"] = self.quality_head(pre_logits_hidden).squeeze(
                    -1
                )
            if quality_target is not None:
                result["quality_target"] = quality_target
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
