"""Canonical training loop for HAGI (data-source-agnostic).

Single source of truth for the forward/backward/step cycle. Absorbs what was
previously three parallel implementations (run_basic / run_fast / run_full):
composite-loss warmup, CE-computed-once contract, EMA, NARS observation, NaN
guard, component + timing logging, and single/sharded checkpointing.

The data source is any zero-arg ``get_batch()`` returning ``(x, y)`` tensors
(or a ``BatchSampler`` yielding ``(batch, targets)`` — see ``batched=True``).
Ampere-specific TF32/bf16 knobs are set once at startup.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING, cast

import torch

import warnings

# Benign, unactionable warnings on this RTX 3070 / Windows setup. Filtered
# centrally here so scripts/train.py, scripts/profile_steps.py, and
# hagi.train.cli all inherit the suppression (they all import this module
# before any CUDA tensor is allocated).
warnings.filterwarnings(
    "ignore",
    message="expandable_segments not supported",
)
warnings.filterwarnings(
    "ignore",
    message=r"Online softmax is disabled",
)

from hagi.train.config import config_from_dict, config_to_dict  # noqa: E402

if TYPE_CHECKING:
    from hagi.model import HAGI


@dataclass
class LoopConfig:
    """All knobs the training loop reads. Built from a training-config dict."""

    max_steps: int = 50000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    precision: str = "bf16"
    gradient_checkpointing: bool = False
    eval_interval: int = 2000
    eval_iters: int = 50
    ckpt_interval: int = 5000
    ckpt_dir: str = "checkpoints"
    log_interval: int = 50
    nars_policy_interval: int = 200
    # Schedule / optimizer
    schedule: str = "cosine"
    cooldown_frac: float = 0.05
    optimizer_kind: str = "adamw"
    magic_norm_max: float = 1.0
    ce_chunk_size: int = 0
    tf32: bool = True
    # EMA
    ema_decay: float = 0.9995
    ema_start_step: int = 1000
    enable_ema: bool = True
    # Composite-loss weights (final targets) and per-term warmup.
    composite_weights: dict[str, float] = field(default_factory=dict)
    w_aux_start: float = 0.0
    w_iso_start: float = 0.0
    w_moe_start: float = 0.0
    w_msa_lb_start: float = 0.0
    aux_warmup_steps: int = 2000
    iso_warmup_steps: int = 5000
    moe_warmup_steps: int = 2000
    loss_warmup_mode: str = "linear"


def _enable_ampere_flags(device: str, tf32: bool) -> None:
    """Set TF32 / bf16-reduction knobs once. No-op off-CUDA."""
    if not (tf32 and device.startswith("cuda") and torch.cuda.is_available()):
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # cudnn autotuner: data.min_seq_len==max_seq_len (fixed T=1024) means every
    # step feeds identical conv/attention shapes, so benchmark=True caches the
    # fastest kernel per shape once instead of the default heuristic. This is the
    # parity gap that made profile_steps.py (which sets it) measure faster than
    # the real loop (which did not). Harmless when shapes vary — just suboptimal.
    torch.backends.cudnn.benchmark = True
    # Ampere fp16/bf16 accumulation reductions — harmless on older HW.
    try:
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    except AttributeError:
        pass


_inductor_decoder_patched = False


def _patch_inductor_decoder() -> None:
    """Make torch._inductor's cl.exe output decode resilient to cp866.

    MSVC ``cl.exe`` on a Russian Windows emits its banner/help in the OEM
    codepage (cp866). Inductor decodes that subprocess output with
    ``locale.getpreferredencoding()`` (cp1251 / utf-8), which raises
    ``UnicodeDecodeError`` (0x98 / 0x8e) and aborts ``torch.compile`` before a
    single kernel is built. Decode as cp866 with ``errors="replace"`` so the
    probe (it only checks whether the binary *is* cl, not the message content)
    succeeds regardless of locale. No-op off-Windows or if inductor is absent.
    """
    global _inductor_decoder_patched
    if _inductor_decoder_patched:
        return
    _inductor_decoder_patched = True
    try:
        import torch._inductor.cpp_builder as _cb  # type: ignore[import-not-found]
    except Exception:
        return
    # (codec, errors) — cp866 matches MSVC's OEM output; replace tolerates any
    # other codepage a non-Russian locale might emit.
    _cb.SUBPROCESS_DECODE_ARGS = ("cp866", "replace")
    # HAGI reads self._step (a buffer) via .item() for the rotor schedule.
    # That .item() graph-breaks the compiled forward; capture_scalar_outputs
    # keeps it inside the graph as a single (cheap) device→host read.
    try:
        import torch._dynamo as _dynamo  # type: ignore[import-not-found]

        _dynamo.config.capture_scalar_outputs = True  # type: ignore[attr-defined]
    except Exception:
        pass


def lr_at(
    step: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_lr_ratio: float = 0.1,
    schedule: str = "cosine",
    cooldown_frac: float = 0.05,
) -> float:
    """Linear warmup then cosine decay (default) or WSD."""
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)
    if str(schedule).lower() == "wsd":
        cooldown_start = int(max_steps * (1.0 - cooldown_frac))
        if step >= cooldown_start:
            cd_progress = (step - cooldown_start) / max(1, max_steps - cooldown_start)
            cd_progress = min(1.0, max(0.0, cd_progress))
            return learning_rate * (
                min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - cd_progress)
            )
        return learning_rate
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return learning_rate * min_lr_ratio + coeff * learning_rate * (1.0 - min_lr_ratio)


def scheduled_weight(
    step: int, start: float, final: float, warmup_steps: int, mode: str = "linear"
) -> float:
    """Ramp a loss weight from ``start`` to ``final`` over ``warmup_steps``."""
    progress = min(1.0, step / max(1, warmup_steps))
    if mode == "cosine":
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (final - start) * progress


def autocast_ctx(precision: str, device: str):
    """Autocast context. ``manual_*`` and fp32 fall back to no autocast so fp32
    master weights are preserved (mixed precision lives in per-op casting only)."""
    if precision in ("fp32", "manual_fp16", "manual_bf16") or not device.startswith(
        "cuda"
    ):
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def update_ema(
    model: torch.nn.Module, model_ema: torch.nn.Module, decay: float
) -> None:
    """In-place Polyak EMA of params + buffer copy."""
    with torch.no_grad():
        params = list(model.parameters())
        ema_params = list(model_ema.parameters())
        if hasattr(torch, "_foreach_mul"):
            ema_device = ema_params[0].device if ema_params else None
            if ema_device == params[0].device:
                torch._foreach_mul_(cast(list[torch.Tensor], ema_params), decay)
                torch._foreach_add_(
                    cast(list[torch.Tensor], ema_params),
                    [p.detach() for p in params],
                    alpha=1.0 - decay,
                )
            else:
                for ema_param, param in zip(ema_params, params, strict=True):
                    ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        else:
            for ema_param, param in zip(ema_params, params, strict=True):
                ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        for ema_buffer, buffer in zip(
            model_ema.buffers(), model.buffers(), strict=True
        ):
            ema_buffer.copy_(buffer)


def magic_norm_clip(
    model: torch.nn.Module, max_norm: float, blade_count: int = 8
) -> torch.Tensor:
    """Per-blade gradient norm clip for GDR grade-decomposed weights."""
    gdr = getattr(model, "gdr", None)
    if gdr is None or max_norm <= 0:
        return torch.tensor(
            0.0, device=next(model.parameters()).device, dtype=torch.float32
        )
    first_param = next((p for p in gdr.parameters() if p.grad is not None), None)
    if first_param is None:
        return torch.tensor(
            0.0, device=next(model.parameters()).device, dtype=torch.float32
        )
    max_norm_t = first_param.new_full((), max_norm, dtype=torch.float32)
    max_seen_t = first_param.new_zeros((), dtype=torch.float32)
    for param in gdr.parameters():
        grad = param.grad
        if grad is None or grad.ndim == 0 or grad.shape[-1] % blade_count != 0:
            continue
        view = grad.view(*grad.shape[:-1], grad.shape[-1] // blade_count, blade_count)
        norms = view.norm(dim=-1, keepdim=True).clamp_min(1e-4).float()
        local_max = norms.max().detach()
        max_seen_t = torch.maximum(max_seen_t, local_max)
        view.mul_((max_norm_t / norms).clamp(max=1.0).detach().to(dtype=view.dtype))
    return max_seen_t


def get_grad_norm(model: torch.nn.Module) -> float:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    norms = torch._foreach_norm(grads)
    total = torch.stack(norms).pow(2).sum()
    return float(total.sqrt().item())


def gpu_util(device: str) -> str:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "n/a"
    try:
        index = torch.device(device).index
        util = torch.cuda.utilization(index if index is not None else 0)
        return f"{util}%"
    except Exception:
        return "n/a"


def _resolve_loss(
    model_output: Any,
    targets: torch.Tensor,
    weights: dict[str, float] | None,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """CE computed once by the model forward (fused or chunked).

    The model returns ``output["loss"]`` as the canonical CE; this only adds the
    aux/iso/moe/quality terms. Composite_loss is reused as the aggregator so the
    per-term device/dtype handling stays in one place.
    """
    from hagi.losses import composite_loss

    logits = None
    precomputed_loss = None
    if isinstance(model_output, dict):
        logits = model_output.get("logits")
        precomputed_loss = model_output.get("loss")

    if weights is None:
        # CE-only path: trust the model's loss (must exist).
        if precomputed_loss is not None:
            return precomputed_loss, {}
        if logits is None:
            raise ValueError(
                "model output has neither 'loss' nor 'logits'; cannot compute CE"
            )
        import torch.nn.functional as F

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return loss, {}

    losses = composite_loss(
        logits,
        targets,
        auxiliary_output=(
            model_output.get("auxiliary_output")
            if isinstance(model_output, dict)
            else None
        ),
        model_output=model_output if isinstance(model_output, dict) else None,
        weights=weights,
        invariant_src=(
            model_output.get("invariant_src")
            if isinstance(model_output, dict)
            else None
        ),
        invariant_tgt=(
            model_output.get("invariant_tgt")
            if isinstance(model_output, dict)
            else None
        ),
        precomputed_loss=precomputed_loss,
        moe_aux_loss=(
            model_output.get("moe_aux_loss") if isinstance(model_output, dict) else None
        ),
        num_moe_layers=(
            model_output.get("num_moe_layers")
            if isinstance(model_output, dict)
            else None
        ),
        msa_aux_loss=(
            model_output.get("msa_aux_loss") if isinstance(model_output, dict) else None
        ),
        chunk_size=chunk_size,
        quality_score=(
            model_output.get("quality_score")
            if isinstance(model_output, dict)
            else None
        ),
        quality_targets=(
            model_output.get("quality_target")
            if isinstance(model_output, dict)
            else None
        ),
    )
    total_loss = losses["L_total"]
    components = {name: value.detach().float() for name, value in losses.items()}
    return total_loss, components


def build_loop_config(
    full_cfg: dict[str, Any], ckpt_dir: str, max_steps: int
) -> LoopConfig:
    """Single source of truth for LoopConfig construction.

    Both ``scripts/train.py`` and ``hagi.train.cli`` call this so the script and
    console entry points cannot drift on schedule/EMA/precision defaults.
    """
    train_cfg = full_cfg.get("training", {})
    ema_cfg = train_cfg.get("ema", {})
    composite_cfg = train_cfg.get("composite_loss")
    composite_weights = dict(composite_cfg) if isinstance(composite_cfg, dict) else {}
    return LoopConfig(
        max_steps=int(max_steps),
        warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
        learning_rate=float(train_cfg.get("learning_rate", 5.0e-4)),
        min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.1)),
        grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        precision=str(train_cfg.get("precision", "bf16")),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        eval_interval=int(train_cfg.get("eval_interval", 2000)),
        eval_iters=int(train_cfg.get("eval_iters", 50)),
        ckpt_interval=int(train_cfg.get("ckpt_interval", 5000)),
        ckpt_dir=str(ckpt_dir),
        log_interval=int(train_cfg.get("log_interval", 50)),
        nars_policy_interval=int(train_cfg.get("nars_policy_interval", 200)),
        schedule=str(train_cfg.get("schedule", "cosine")),
        cooldown_frac=float(train_cfg.get("cooldown_frac", 0.05)),
        optimizer_kind=str(train_cfg.get("optimizer", "adamw")).lower(),
        magic_norm_max=float(train_cfg.get("magic_norm_max", 1.0)),
        ce_chunk_size=int(full_cfg.get("model", {}).get("ce_chunk_size", 0)),
        tf32=True,
        ema_decay=float(ema_cfg.get("decay", 0.9995)),
        ema_start_step=int(ema_cfg.get("start_step", 1000)),
        enable_ema="ema" in train_cfg or "ema_decay" in train_cfg,
        composite_weights=composite_weights,
        w_aux_start=float(train_cfg.get("w_aux_start", 0.0)),
        w_iso_start=float(train_cfg.get("w_iso_start", 0.0)),
        w_moe_start=float(train_cfg.get("w_moe_start", 0.0)),
        w_msa_lb_start=float(train_cfg.get("w_msa_lb_start", 0.0)),
        aux_warmup_steps=int(
            train_cfg.get("aux_warmup_steps", train_cfg.get("aux_warmup", 2000))
        ),
        iso_warmup_steps=int(
            train_cfg.get("iso_warmup_steps", train_cfg.get("iso_warmup", 5000))
        ),
        moe_warmup_steps=int(
            train_cfg.get("moe_warmup_steps", train_cfg.get("moe_warmup", 2000))
        ),
        loss_warmup_mode=str(train_cfg.get("loss_warmup_mode", "linear")),
    )


def train(
    model: HAGI,
    optimizer,
    get_batch: Any,
    cfg: LoopConfig,
    device: str = "cpu",
    eval_get_batch: Callable[..., Any] | None = None,
    on_log: Callable[[dict[str, Any]], None] | None = None,
    start_step: int = 0,
    session_steps: int | None = None,
    on_checkpoint: Callable[[str], None] | None = None,
    on_eval: Callable[[int], None] | None = None,
    model_ema: torch.nn.Module | None = None,
    batched: bool = False,
    use_prefix_lm: bool = False,
    to_device_fn: Callable[..., Any] | None = None,
    apply_prefix_mask_fn: Callable[..., Any] | None = None,
    sequential_state_fn: Callable[[], dict[str, int] | None] | None = None,
) -> float:
    """Run the canonical training loop. Returns the final training loss.

    Args:
        get_batch: zero-arg callable returning ``(x, y)`` (``batched=False``) or
            a next-able iterator/dataloader yielding ``(batch, targets)`` where
            ``batch`` may be a ``PrefixLMBatch`` (``batched=True``).
        model_ema: optional EMA model maintained in-place; created from a deep
            copy of ``model`` if ``cfg.enable_ema`` and None passed.
        to_device_fn / apply_prefix_mask_fn: optional hooks for the batched
            PrefixLM path; unused when ``batched=False``.
        sequential_state_fn: optional zero-arg callable returning the
            SequentialCyclingIterator position (current_idx/current_cycle) to
            persist into checkpoints; None disables persistence. The caller
            (scripts/train.py) passes a closure over its dataloader.
    """
    _enable_ampere_flags(device, cfg.tf32)
    model.to(device)
    model.train()
    if hasattr(model.cfg, "gradient_checkpointing"):
        model.cfg.gradient_checkpointing = cfg.gradient_checkpointing

    # Opt-in whole-model compile (model cfg.compile). Forward passes go through
    # `run_model`; checkpoints keep using `model` so state_dict keys never get
    # the `_orig_mod.` prefix.
    run_model = model
    if getattr(getattr(model, "cfg", None), "compile", False) and device.startswith(
        "cuda"
    ):
        if hasattr(torch, "compile"):
            _patch_inductor_decoder()
            # dynamic=True: variable-length training (data.min_seq_len=256 ->
            # max_seq_len=1024) draws a fresh T every batch, so Dynamo would
            # specialize a guard on targets.shape[1] and hit recompile_limit(8)
            # -> eager fallback (wasted step-0 compile). dynamic shapes let one
            # graph cover the whole T range with no recompiles.
            # mode left default (NOT "max-autotune"): the inductor Triton autotuner
            # segfaults (0xC0000005 access violation) in bf16 backward on this
            # Windows/torch build — the autotuned GEMM kernel is ABI-incompatible.
            # default mode uses the stable eager-fallback kernels.
            run_model = torch.compile(model, dynamic=True)

    precision = cfg.precision
    use_scaler = precision == "fp16" and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    manual_lowprec = precision in ("manual_fp16", "manual_bf16") and device.startswith(
        "cuda"
    )
    if manual_lowprec:
        cast_dtype = torch.float16 if precision == "manual_fp16" else torch.bfloat16
        model.to(cast_dtype)
        if model_ema is not None:
            model_ema.to(cast_dtype)
        print(
            f"Using manual {precision}: model cast to {cast_dtype}, no autocast "
            "(fp32 master weights lost)"
        )

        # Realign optimizer moment buffers to the (now low-precision) param
        # dtype. Fresh run: state is lazy (empty) -> no-op. On resume,
        # scripts/train.py loads the saved optimizer state BEFORE this cast:
        # load_state_dict upconverts the bf16 moment buffers to the model's
        # then-fp32 params, then this cast flips the params to bf16 but leaves
        # the upconverted fp32 moments alone, so the fused AdamW kernel aborts:
        #   "params, grads, exp_avgs, and exp_avg_sqs must have same dtype".
        # Cast only the N-dim moment tensors (exp_avg, exp_avg_sq, Muon's
        # momentum_buffer) to the param dtype. The 0-dim ``step`` counter is
        # left untouched: capturable=True fused AdamW keeps it fp32 on purpose,
        # and casting it to bf16 corrupts the increment at large step values
        # (bf16 ulp ~8 near 2000 -> step += 1 rounds back -> NaN downstream).
        if optimizer is not None:
            sub_opts = getattr(optimizer, "optimizers", [optimizer])
            for opt in sub_opts:
                for group in opt.param_groups:
                    for p in group["params"]:
                        st = opt.state.get(p)
                        if not st:
                            continue
                        pdt = p.data.dtype
                        for _k, v in st.items():
                            if (
                                isinstance(v, torch.Tensor)
                                and v.dim() > 0
                                and v.dtype != pdt
                            ):
                                st[_k] = v.to(dtype=pdt, device=p.data.device)

    # EMA: eager init to avoid copy.deepcopy inside the hot loop.
    if cfg.enable_ema and model_ema is None:
        import copy

        model_ema = copy.deepcopy(model).to(device)
        model_ema.eval()
        for param in model_ema.parameters():
            param.requires_grad_(False)
    if model_ema is not None:
        model_ema.eval()
        for param in model_ema.parameters():
            param.requires_grad_(False)

    # Per-group base LR so the schedule scales each group from its own start.
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    # NARS HRM controller (no-op when model.use_nars is False).
    nars_hrm = getattr(model, "nars_hrm", None)

    composite_weights = cfg.composite_weights or None
    final_aux = composite_weights.get("w_aux", 0.1) if composite_weights else 0.0
    final_iso = composite_weights.get("w_iso", 0.01) if composite_weights else 0.0
    final_moe = composite_weights.get("w_moe", 0.0) if composite_weights else 0.0
    final_msa_lb = composite_weights.get("w_msa_lb", 0.0) if composite_weights else 0.0

    last_loss = float("nan")
    end = (
        cfg.max_steps
        if session_steps is None
        else min(cfg.max_steps, start_step + session_steps)
    )
    start_time = time.perf_counter()
    tokens_since_log = 0
    last_log_time = start_time
    # Total tokens processed across all forward micro-batches this run.
    # Accumulated from tokens.numel() per accum step so the final throughput
    # line reports real tokens/sec (was 0: the old (end-start)*grad_accum
    # expression counted optimiser steps, not tokens).
    total_tokens_seen = 0
    data_iter: Any = iter(get_batch) if batched else None

    for step in range(start_step, end):
        if cfg.optimizer_kind == "schedule-free-adamw":
            lr = cfg.learning_rate
        else:
            lr = lr_at(
                step,
                cfg.max_steps,
                cfg.warmup_steps,
                cfg.learning_rate,
                cfg.min_lr_ratio,
                schedule=cfg.schedule,
                cooldown_frac=cfg.cooldown_frac,
            )
        ratio = lr / max(cfg.learning_rate, 1e-12)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * ratio

        # Composite-loss weight warmup (ramps from *_start to final target).
        effective_weights = None
        if composite_weights is not None:
            effective_weights = dict(composite_weights)
            effective_weights["w_aux"] = scheduled_weight(
                step,
                cfg.w_aux_start,
                final_aux,
                cfg.aux_warmup_steps,
                cfg.loss_warmup_mode,
            )
            effective_weights["w_iso"] = scheduled_weight(
                step,
                cfg.w_iso_start,
                final_iso,
                cfg.iso_warmup_steps,
                cfg.loss_warmup_mode,
            )
            effective_weights["w_moe"] = scheduled_weight(
                step,
                cfg.w_moe_start,
                final_moe,
                cfg.moe_warmup_steps,
                cfg.loss_warmup_mode,
            )
            effective_weights["w_msa_lb"] = scheduled_weight(
                step,
                cfg.w_msa_lb_start,
                final_msa_lb,
                cfg.moe_warmup_steps,
                cfg.loss_warmup_mode,
            )

        optimizer.zero_grad(set_to_none=True)
        accum_loss_tensor: torch.Tensor | None = None
        accum_components: dict[str, torch.Tensor] = {}
        need_components = cfg.log_interval > 0 and step % cfg.log_interval == 0
        backward_count = 0
        t_forward = 0.0
        t_backward = 0.0

        for _ in range(cfg.grad_accum_steps):
            if batched:
                try:
                    batch, targets = next(data_iter)  # type: ignore[arg-type]
                except StopIteration:
                    data_iter = iter(get_batch)  # type: ignore[arg-type]
                    batch, targets = next(data_iter)  # type: ignore[arg-type]
                if to_device_fn is not None:
                    batch = to_device_fn(batch, device, device.startswith("cuda"))
                targets = targets.to(device, non_blocking=device.startswith("cuda"))
                if apply_prefix_mask_fn is not None:
                    targets = apply_prefix_mask_fn(targets, batch)
                # batch may be a PrefixLMBatch exposing .tokens, else a tensor.
                tokens = batch.tokens if hasattr(batch, "tokens") else batch
            else:
                x, y = get_batch()
                tokens, targets = x, y

            t_fwd_start = time.perf_counter()
            with autocast_ctx(precision, device):
                output = run_model(
                    tokens,
                    targets=targets,
                    training_mode=effective_weights is not None,
                    weights=effective_weights,
                )
                # Fast path: skip composite_loss aggregation when only CE needed.
                if (
                    not need_components
                    and effective_weights is not None
                    and effective_weights.get("w_aux", 0) == 0
                    and effective_weights.get("w_iso", 0) == 0
                    and effective_weights.get("w_moe", 0) == 0
                    and effective_weights.get("w_msa_lb", 0) == 0
                    and effective_weights.get("w_quality", 0) == 0
                    and isinstance(output, dict)
                    and output.get("loss") is not None
                ):
                    loss = output["loss"]
                    components: dict[str, torch.Tensor] = {}
                else:
                    loss, components = _resolve_loss(
                        output, targets, effective_weights, cfg.ce_chunk_size
                    )
                raw_loss = loss.detach().float()
                loss = loss / cfg.grad_accum_steps
                if device.startswith("cuda") and need_components:
                    torch.cuda.synchronize()
                t_forward += time.perf_counter() - t_fwd_start

            if not torch.isfinite(loss).all():
                if need_components:
                    print(
                        f"WARNING: non-finite loss at step {step}; skipping accum step"
                    )
                continue

            t_bwd_start = time.perf_counter()
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            t_backward += time.perf_counter() - t_bwd_start
            backward_count += 1
            accum_loss_tensor = (
                raw_loss if accum_loss_tensor is None else accum_loss_tensor + raw_loss
            )
            if components and need_components:
                for name, value in components.items():
                    prev = accum_components.get(name)
                    accum_components[name] = value if prev is None else prev + value
            tokens_since_log += tokens.numel()
            total_tokens_seen += tokens.numel()

        if backward_count == 0:
            last_loss = float("nan")
            if need_components:
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-9)
                print(
                    f"step {step:6d} | loss nan | lr {lr:.2e} | "
                    f"tokens/sec {tokens_since_log / elapsed:.0f} | gpu_util {gpu_util(device)} | SKIPPED"
                )
                tokens_since_log = 0
                last_log_time = now
            continue

        if device.startswith("cuda") and need_components:
            torch.cuda.synchronize()
        t_opt_start = time.perf_counter()
        if use_scaler:
            scaler.unscale_(optimizer)

        grad_norm_val = 0.0
        if cfg.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            )
            grad_norm_val = float(grad_norm.item())
            if need_components and (
                not math.isfinite(grad_norm_val)
                or grad_norm_val > 100.0
                or (0.0 < grad_norm_val < 1e-6)
            ):
                print(f"WARNING: extreme grad_norm {grad_norm_val:.2e} at step {step}")
        else:
            grad_norm_val = get_grad_norm(model)

        magic_grad = magic_norm_clip(model, cfg.magic_norm_max)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        t_opt = time.perf_counter() - t_opt_start

        if step >= cfg.ema_start_step and model_ema is not None:
            update_ema(model, model_ema, cfg.ema_decay)

        # NARS HRM: observe every step (cheap), re-apply policy on a coarse
        # interval to avoid per-step cycle churn destabilising training.
        if nars_hrm is not None:
            if cfg.grad_clip <= 0:
                grad_norm_val = get_grad_norm(model)
            # Sanitize: a non-finite loss would poison the truth revision
            # (_clamp01(nan)=nan -> _build_policy int() ValueError crash) and
            # kill the run. Feed 0.0 so observation stays well-defined.
            obs_loss = (
                float(accum_loss_tensor.item() / cfg.grad_accum_steps)
                if accum_loss_tensor is not None
                else 0.0
            )
            obs_grad = grad_norm_val if math.isfinite(grad_norm_val) else 0.0
            nars_hrm.observe_train_step(
                obs_loss if math.isfinite(obs_loss) else 0.0,
                obs_grad,
            )
            if step % cfg.nars_policy_interval == 0:
                policy = nars_hrm.resolve_policy()
                hrm = getattr(model, "hrm", None)
                if hrm is not None:
                    nars_hrm.apply_policy(policy, hrm)

        # Track running loss every step (not only on log steps) so the return
        # value is real even when the final step isn't a log step.
        last_loss = (
            float((accum_loss_tensor / cfg.grad_accum_steps).cpu().item())
            if accum_loss_tensor is not None
            else float("nan")
        )
        if need_components:
            last_components = {
                name: (value / cfg.grad_accum_steps).item()
                for name, value in accum_components.items()
            }
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-9)
            tok_per_sec = tokens_since_log / elapsed
            component_text = ""
            if last_components:
                component_text = " | " + " | ".join(
                    f"{name} {value:.4f}" for name, value in last_components.items()
                )
            mem_text = ""
            if device.startswith("cuda"):
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                mem_text = f" | mem_allocated {allocated:.2f}GB | mem_reserved {reserved:.2f}GB"
            eval_tag = (
                "ema"
                if (step >= cfg.ema_start_step and model_ema is not None)
                else "model"
            )
            metrics = {
                "step": step,
                "loss": last_loss,
                "lr": lr,
                "grad_norm": grad_norm_val,
                "tokens_per_sec": tok_per_sec,
                "eval_model": eval_tag,
            }
            metrics.update(last_components)
            if on_log:
                on_log(metrics)
            else:
                print(
                    f"step {step:6d} | loss {last_loss:.4f}{component_text} | lr {lr:.2e}"
                    f" | grad_norm {grad_norm_val:.2e} | magic_norm {magic_grad.item():.4f}"
                    f" | tokens/sec {tok_per_sec:.0f} | gpu_util {gpu_util(device)}{mem_text}"
                    f" | fwd {t_forward*1000:.1f}ms | bwd {t_backward*1000:.1f}ms | opt {t_opt*1000:.1f}ms"
                )
            tokens_since_log = 0
            last_log_time = now

        if cfg.eval_interval > 0 and step > 0 and step % cfg.eval_interval == 0:
            if on_eval is not None:
                # Full eval harness (ppl/acc over a held-out loader) owned by
                # the caller; picks the EMA weights when available.
                on_eval(step)
            elif eval_get_batch is not None:
                val = estimate_loss(
                    model, eval_get_batch, cfg.eval_iters, device, precision
                )
                print(f"step {step:6d} | val_loss {val:.4f}")

        if cfg.ckpt_interval > 0 and step > 0 and step % cfg.ckpt_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                step,
                cfg.ckpt_dir,
                ema_state=(model_ema.state_dict() if model_ema is not None else None),
                on_checkpoint=on_checkpoint,
                sequential_state=(
                    sequential_state_fn() if sequential_state_fn else None
                ),
            )

    if session_steps is not None and on_checkpoint is not None:
        save_checkpoint(
            model,
            optimizer,
            end,
            cfg.ckpt_dir,
            ema_state=(model_ema.state_dict() if model_ema is not None else None),
            on_checkpoint=on_checkpoint,
            sequential_state=(
                sequential_state_fn() if sequential_state_fn else None
            ),
        )
    elif session_steps is not None:
        save_checkpoint(
            model,
            optimizer,
            end,
            cfg.ckpt_dir,
            ema_state=(model_ema.state_dict() if model_ema is not None else None),
            sequential_state=(
                sequential_state_fn() if sequential_state_fn else None
            ),
        )

    if total_tokens_seen > 0:
        total_elapsed = max(time.perf_counter() - start_time, 1e-9)
        print(
            f"final_loss {last_loss:.4f} | avg_tokens/sec "
            f"{total_tokens_seen / total_elapsed:.0f} | "
            f"steps {(end - start_step)} | tokens {total_tokens_seen:,}"
        )
    return last_loss


@torch.no_grad()
def estimate_loss(
    model: HAGI, get_batch: Callable[..., Any], iters: int, device: str, precision: str
) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch()
        with autocast_ctx(precision, device):
            _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(
    model: HAGI,
    optimizer,
    step: int,
    ckpt_dir: str,
    ema_state: dict[str, Any] | None = None,
    on_checkpoint: Callable[[str], None] | None = None,
    sequential_state: dict[str, int] | None = None,
):
    """Write a checkpoint with config, optimizer, and optional EMA state."""
    out = Path(ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"step-{step:08d}.pt"
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
        "config": config_to_dict(model.cfg),
        "optimizer": optimizer.state_dict(),
    }
    if sequential_state is not None:
        payload["sequential_state"] = sequential_state
    if ema_state is not None:
        payload["model_ema"] = {
            name: value.detach().cpu() for name, value in ema_state.items()
        }
    if hasattr(model, "msa_registry") and model.msa_registry is not None:
        payload["msa_registry"] = model.msa_registry.state_dict()
    if hasattr(model, "nars_hrm") and model.nars_hrm is not None:
        payload["nars_hrm"] = model.nars_hrm.state_dict()
    if hasattr(model, "nars_hdim") and model.nars_hdim is not None:
        payload["nars_hdim"] = model.nars_hdim.state_dict()
    if hasattr(model, "nars_msa") and model.nars_msa is not None:
        payload["nars_msa"] = model.nars_msa.state_dict()
    torch.save(payload, path)
    print(f"checkpoint -> {path}")
    if on_checkpoint is not None:
        on_checkpoint(str(path))


def _resume_load_state_dict(
    model: HAGI,
    ckpt_state: dict[str, Any],
    new_cfg: Any,
    old_cfg: Any,
) -> None:
    """Load ``ckpt_state`` into ``model`` with ``strict=False`` and log the gaps.

    Used on resume when the caller overrides the architecture via
    ``model_cfg_override``: shared params load from the ckpt, new-architecture
    params fresh-init, dropped ckpt params are skipped. The dangerous case is
    params present in both with different shapes (e.g. hrm weights when
    ``hrm_l_cycles`` changed): ``strict=False`` skips them silently, so they are
    surfaced as a WARNING list so the user knows which params fresh-init.
    """
    model_state = model.state_dict()
    # Detect shape mismatches BEFORE load: keys present in both with different
    # shapes are silently skipped by strict=False, so surface them explicitly.
    shape_mismatches = [
        k
        for k in ckpt_state
        if k in model_state
        and hasattr(ckpt_state[k], "shape")
        and ckpt_state[k].shape != model_state[k].shape
    ]
    missing_keys, unexpected_keys = model.load_state_dict(ckpt_state, strict=False)
    print(
        f"[resume] loaded weights strict=False "
        f"(override cfg vs ckpt config: "
        f"{type(new_cfg).__name__} over {type(old_cfg).__name__ if old_cfg is not None else 'n/a'})"
    )
    print(
        f"[resume] missing keys (fresh-init): {len(missing_keys)}, "
        f"unexpected keys (dropped ckpt params): {len(unexpected_keys)}, "
        f"shape-mismatch keys (skipped, fresh-init): {len(shape_mismatches)}"
    )
    if shape_mismatches:
        preview = ", ".join(shape_mismatches[:20])
        more = (
            f" ... (+{len(shape_mismatches) - 20} more)"
            if len(shape_mismatches) > 20
            else ""
        )
        print(f"[resume] WARNING shape-mismatch keys: {preview}{more}")


def load_checkpoint(
    path: str,
    device: str = "cpu",
    optimizer=None,
    load_ema: bool = False,
    use_ema: bool = False,
    model_cfg_override: Any = None,
) -> tuple[HAGI, int, dict[str, Any] | None]:
    """Rebuild a HAGI model from a checkpoint.

    All tensors are loaded to CPU first (``map_location="cpu"``) to avoid
    pinning the full checkpoint — model + optimizer + EMA — in VRAM during
    resume.  ``model.to(device)`` then moves only the model weights to the
    target device.

    Args:
        model_cfg_override: when not None, build the model from this config
            instead of the (possibly stale) one baked into the checkpoint, and
            load the saved weights with ``strict=False`` so architectural
            changes (hrm_l_cycles, use_quality_head, hdim_heads, ...) carry
            forward — shared params load from the ckpt, new/dropped params
            fresh-init or are skipped. A shape-mismatch WARNING list is printed
            for params present in both whose shapes differ (silently skipped by
            ``strict=False`` but the dangerous case: e.g. hrm weights when
            hrm_l_cycles changed). When None (chat.py / eval callers), keep the
            exact current behavior: build from the ckpt config, ``strict=True``.

    Returns:
        (model, step, ema_state | None)
    """
    from hagi.model import HAGI

    p = Path(path)
    # Sharded checkpoint directory (model.pt, optimizer.pt, ema.pt, meta.pt)
    if p.is_dir() and (p / "model.pt").exists():
        meta = (
            torch.load(p / "meta.pt", map_location=device, weights_only=True)
            if (p / "meta.pt").exists()
            else {}
        )
        if model_cfg_override is not None:
            cfg = model_cfg_override
            model = HAGI(cfg)
        else:
            cfg = config_from_dict(meta["config"])
            model = HAGI(cfg)
        state_dict = torch.load(p / "model.pt", map_location="cpu", weights_only=True)
        if any(k.startswith("hrm._orig_mod.") for k in state_dict):
            state_dict = {
                k.replace("hrm._orig_mod.", "hrm.", 1): v for k, v in state_dict.items()
            }
        for key in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
            state_dict.pop(key, None)
        if model_cfg_override is not None:
            _resume_load_state_dict(model, state_dict, cfg, meta.get("config"))
        else:
            model.load_state_dict(state_dict)
        del state_dict
        model.to(device)
        ema_state = None
        if (use_ema or load_ema) and (p / "ema.pt").exists():
            ema_state = torch.load(p / "ema.pt", map_location="cpu", weights_only=True)
            if use_ema:
                model.load_state_dict(ema_state)
                ema_state = None
        step = int(meta.get("step", 0))
        del meta
        import gc

        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model, step, ema_state

    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
        state.pop(key, None)
    if model_cfg_override is not None:
        cfg = model_cfg_override
    else:
        cfg = config_from_dict(state["config"])
    model = HAGI(cfg)

    if state.get("_inference_opt"):
        from hagi.model.inference_opt import (
            fold_rmsnorm_into_weights,
            precompute_rope_tables,
            repack_qkv_for_contiguous,
        )

        max_seq_len = state.get("_rope_max_seq_len", cfg.transformer.max_seq_len)
        fold_rmsnorm_into_weights(model)
        repack_qkv_for_contiguous(model)
        precompute_rope_tables(model, max_seq_len)

    if model_cfg_override is not None:
        _resume_load_state_dict(model, state["model"], cfg, state.get("config"))
    else:
        model.load_state_dict(state["model"])
    model.to(device)

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    if (
        hasattr(model, "msa_registry")
        and model.msa_registry is not None
        and "msa_registry" in state
    ):
        model.msa_registry.load_state_dict(state["msa_registry"])
    if (
        hasattr(model, "nars_hrm")
        and model.nars_hrm is not None
        and "nars_hrm" in state
    ):
        model.nars_hrm.load_state_dict(state["nars_hrm"])
    if (
        hasattr(model, "nars_hdim")
        and model.nars_hdim is not None
        and "nars_hdim" in state
    ):
        model.nars_hdim.load_state_dict(state["nars_hdim"])
    if (
        hasattr(model, "nars_msa")
        and model.nars_msa is not None
        and "nars_msa" in state
    ):
        model.nars_msa.load_state_dict(state["nars_msa"])

    ema_state = state.get("model_ema") if (use_ema or load_ema) else None
    step = int(state.get("step", 0))
    if use_ema and ema_state is not None:
        model.load_state_dict(ema_state)
        ema_state = None
    del state
    import gc

    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, step, ema_state
