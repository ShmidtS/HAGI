from __future__ import annotations

import argparse
import gc
import math
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any, cast

# Expandable segments must be set before the CUDA caching allocator inits
# (i.e. before any CUDA tensor is allocated). Without it, SequentialCycling's
# per-dataset KV/registry allocations fragment the 8GB pool and
# memory_reserved climbs past the 7.4GB budget (observed 8.09GB at step 5
# vs a stable 7.15GB with this flag). No-op off-CUDA / already-set.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from hagi.data import (
    MemmapDataset,
    PrefixLMBatch,
    SequentialCyclingIterator,
    create_prefix_lm_batch,
    get_mixed_memmap_dataloader,
    get_sft_dataloader,
)
from hagi.data.tokenizer import TokenizerWrapper
from hagi.model import HAGI
from hagi.train.config import config_from_dict
from hagi.train.loop import (
    LoopConfig,
    autocast_ctx,
    train,
)
from hagi.train.optim import build_optimizer
from hagi.utils import _load_yaml as load_yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "rtx3070_canonical.yaml"
DEFAULT_CKPT_DIR = ROOT / "checkpoints" / "rtx3070"
DEFAULT_DATA_DIR = ROOT / "data"


def _resolve_max_steps(
    tcfg: dict[str, Any], data_cfg: dict[str, Any], batch_size: int, block_size: int
) -> int:
    """Derive max_steps from train_tokens if set, else fall back to max_steps."""
    grad_accum = tcfg.get("grad_accum_steps", 1)
    tokens_per_step = batch_size * grad_accum * block_size
    train_tokens = data_cfg.get("train_tokens")
    if train_tokens:
        steps = math.ceil(train_tokens / tokens_per_step)
        print(
            f"max_steps={steps} derived from train_tokens={train_tokens:,} "
            f"({tokens_per_step:,} tokens/step)"
        )
        return steps
    return tcfg.get("max_steps", 50000)


def resolve_train_path(
    cfg: dict[str, Any], train_path_override: Path | None, data_dir: Path | None = None
) -> Path:
    if data_dir is None:
        data_dir = train_path_override
        train_path_override = None
    if train_path_override is not None:
        return Path(train_path_override)

    data_cfg = cfg.get("data", {})
    configured = data_cfg.get("train_path") or data_cfg.get("path")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"train path not found: {path}")
        return path
    if data_dir is None:
        raise ValueError(
            "data_dir is required when no train_path override or configured path"
        )
    bin_files = sorted(data_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"no memmap .bin files found in {data_dir}")
    return bin_files[0]


def print_model_size(model: HAGI) -> None:
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp16_gb = params * 2 / 1024**3
    adamw_gb = params * 12 / 1024**3
    print(f"model parameters: total={params:,} trainable={trainable:,}")
    print(
        f"estimated VRAM: params_fp16={fp16_gb:.2f}GB adamw_training_state~={adamw_gb:.2f}GB"
    )


def print_vram_usage() -> None:
    if not torch.cuda.is_available():
        print("VRAM unavailable: CUDA is not available")
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(
        f"VRAM after model creation: allocated={allocated:.2f}GB reserved={reserved:.2f}GB"
    )


def synthetic_batcher(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: str,
    generator: torch.Generator,
):
    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(
            vocab_size, (batch_size, seq_len), generator=generator, device=device
        )
        y = torch.randint(
            vocab_size, (batch_size, seq_len), generator=generator, device=device
        )
        return x, y

    return get_batch


def memmap_batcher(
    path: Path,
    batch_size: int,
    seq_len: int,
    device: str,
    dtype: str,
    generator: torch.Generator,
):
    dataset = MemmapDataset(path, block_size=seq_len, dtype=dtype, preload=True)
    if len(dataset) <= 0:
        raise ValueError(f"memmap dataset is too small for seq_len={seq_len}: {path}")

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(
            len(dataset), (batch_size,), generator=generator
        ).tolist()
        xs, ys = zip(*(dataset[index] for index in indices), strict=True)
        x = torch.from_numpy(np.stack(xs)).to(
            device=device, dtype=torch.long, non_blocking=device.startswith("cuda")
        )
        y = torch.from_numpy(np.stack(ys)).to(
            device=device, dtype=torch.long, non_blocking=device.startswith("cuda")
        )
        return x, y

    return get_batch


def build_basic_batcher(
    cfg: dict[str, Any],
    device: str,
    train_path: Path | None,
    data_dir: Path,
    seq_len: int | None,
):
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    vocab_size = int(model_cfg.get("vocab_size", 49152))
    batch_size = int(train_cfg.get("batch_size", 1))
    seq_len = int(
        seq_len
        if seq_len is not None
        else data_cfg.get(
            "max_seq_len", model_cfg.get("transformer", {}).get("max_seq_len", 2048)
        )
    )
    seed = int(train_cfg.get("seed", 42))

    if train_path is None:
        configured_path = data_cfg.get("train_path") or data_cfg.get("path")
        train_path = Path(configured_path) if configured_path else None
    if train_path is None and data_dir.exists():
        bin_files = sorted(data_dir.glob("*.bin"))
        train_path = bin_files[0] if bin_files else None
    if train_path is not None and train_path.exists():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return memmap_batcher(
            train_path,
            batch_size,
            seq_len,
            device,
            data_cfg.get("dtype", "uint16"),
            generator,
        )
    warnings.warn(
        "no memmap .bin data found; falling back to synthetic data",
        RuntimeWarning,
        stacklevel=2,
    )
    generator = torch.Generator(device=device if device.startswith("cuda") else "cpu")
    generator.manual_seed(seed)
    return synthetic_batcher(vocab_size, batch_size, seq_len, device, generator)


def load_resume(model: HAGI, resume: Path, device: str) -> int:
    if resume.is_dir():
        model_path = resume / "model.pt"
        if model_path.exists():
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
        meta_path = resume / "meta.pt"
        if meta_path.exists():
            meta = torch.load(meta_path, map_location=device, weights_only=True)
            return int(meta.get("step", 0))
        return 0
    state = torch.load(resume, map_location=device, weights_only=True)
    if "model" in state:
        model.load_state_dict(state["model"])
        return int(state.get("step", 0))
    model.load_state_dict(state)
    return 0


def maybe_disable_gradient_checkpointing(
    model: HAGI,
    device: str,
    batch_size: int = 4,
    seq_len: int = 1024,
    target_batch: int = 8,
    safety: float = 1.0,
    headroom: float = 0.85,
    explicit: bool = False,
) -> None:
    """Disable gradient checkpointing if a dry-run shows enough memory headroom.

    Creates a fresh model from the same config, runs a single forward+backward
    with a half-size batch, extrapolates to the target training batch size,
    and disables checkpointing only when the estimated peak stays below
    ``headroom * total_memory``.

    Skipped entirely when ``explicit=True`` (the user set
    ``training.gradient_checkpointing`` in config) — the probe builds a full
    second model in VRAM, and probing the no-checkpoint path on a model that
    needs checkpointing OOMs and leaves CUDA fragmented for the real run.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    if not getattr(model.cfg, "gradient_checkpointing", False):
        return
    if explicit:
        total = torch.cuda.get_device_properties(device).total_memory
        print(
            f"Kept gradient checkpointing (explicit config; {total / 1024**3:.1f}GB GPU)"
        )
        return

    total = torch.cuda.get_device_properties(device).total_memory
    test_model = HAGI(model.cfg).to(device)
    param_dtype = next(model.parameters()).dtype
    if param_dtype != torch.float32:
        test_model = test_model.to(param_dtype)
    test_model.cfg.gradient_checkpointing = False
    test_model.train()
    try:
        vocab_size = getattr(model.cfg, "vocab_size", 49152)
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            output = test_model(x, targets=y, training_mode=True)
            loss = output["loss"] if isinstance(output, dict) else output[1]
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
        scale = target_batch / batch_size * safety
        est_peak = peak * scale
        if est_peak > total * headroom:
            print(
                f"Kept gradient checkpointing (est. peak {est_peak / 1024**3:.1f}GB > "
                f"{headroom * 100:.0f}% of {total / 1024**3:.1f}GB)"
            )
        else:
            model.cfg.gradient_checkpointing = False
            print(
                f"Disabled gradient checkpointing (est. peak {est_peak / 1024**3:.1f}GB <= "
                f"{headroom * 100:.0f}% of {total / 1024**3:.1f}GB)"
            )
    except Exception:
        pass
    finally:
        del test_model
        torch.cuda.empty_cache()


def maybe_compile(model: HAGI, device: str) -> Any:
    if (
        hasattr(model.cfg, "compile")
        and model.cfg.compile
        and device.startswith("cuda")
    ):
        # MSVC cl.exe on a Russian Windows emits cp866; inductor's cp1251/utf-8
        # decode of `cl /help` aborts torch.compile before any kernel builds.
        from hagi.train.loop import _patch_inductor_decoder

        _patch_inductor_decoder()
        print("torch.compile enabled (mode=default)")
        return torch.compile(model, mode="default")
    return model


def to_device(batch: Any, device: str, non_blocking: bool) -> Any:
    if isinstance(batch, PrefixLMBatch):
        return PrefixLMBatch(
            tokens=batch.tokens.to(device, non_blocking=non_blocking),
            mask=batch.mask.to(device, non_blocking=non_blocking),
            partition=batch.partition.to(device, non_blocking=non_blocking),
        )
    if isinstance(batch, tuple):
        return tuple(item.to(device, non_blocking=non_blocking) for item in batch)
    return batch.to(device, non_blocking=non_blocking)


def apply_prefix_mask(targets: torch.Tensor, batch: Any) -> torch.Tensor:
    if not isinstance(batch, PrefixLMBatch):
        return targets
    positions = torch.arange(targets.size(1), device=targets.device).unsqueeze(0)
    return targets.masked_fill(positions < batch.partition.unsqueeze(1), -100)


def prefix_lm_collate(
    batch: list[Any], seq_len: int
) -> tuple[PrefixLMBatch, torch.Tensor]:
    array = np.stack([np.asarray(item, dtype=np.int64) for item in batch])
    tokens = array[:, :-1]
    targets = torch.as_tensor(array[:, 1:], dtype=torch.long)
    prefix_batch = create_prefix_lm_batch(tokens.tolist(), seq_len)
    return prefix_batch, targets


def _shift_collate(
    batch: list[Any], pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate variable-length samples, right-padding to the batch max.

    Inputs shorter than the batch max are padded with ``pad_id``; their targets
    are padded with ``ignore_index = -100`` so the loss (ignore_index=-100) and
    accuracy skip padded positions. Equal-length batches take the fast path
    (single np.stack, no padding).
    """
    items = [np.asarray(item, dtype=np.int64) for item in batch]
    lens = np.array([it.shape[0] for it in items])
    max_len = int(lens.max())
    if int(lens.min()) == max_len:
        array = np.stack(items)
        x = array[:, :-1]
        y = array[:, 1:]
        return torch.as_tensor(x, dtype=torch.long), torch.as_tensor(
            y, dtype=torch.long
        )
    # Variable lengths: pad each row to max_len. Output length is max_len-1.
    out_len = max_len - 1
    x = np.full((len(items), out_len), pad_id, dtype=np.int64)
    y = np.full((len(items), out_len), -100, dtype=np.int64)
    for i, it in enumerate(items):
        n = it.shape[0] - 1  # valid (input,target) pairs from this sample
        if n > 0:
            x[i, :n] = it[:-1]
            y[i, :n] = it[1:]
    return torch.as_tensor(x, dtype=torch.long), torch.as_tensor(y, dtype=torch.long)


def resolve_mix_paths(
    data_cfg: dict[str, Any], data_dir: Path
) -> list[tuple[Path, float]]:
    mix_paths = data_cfg.get("mix_paths", [])
    resolved: list[tuple[Path, float]] = []
    for entry in mix_paths:
        path = Path(entry["path"])
        if not path.exists():
            path = data_dir / path
        resolved.append((path, float(entry["weight"])))
    return resolved


def _resolve_sequential_entries(
    data_cfg: dict[str, Any], data_dir: Path
) -> list[dict[str, Any]]:
    """Resolve mix_paths entries with names for sequential cycling."""
    entries: list[dict[str, Any]] = []
    for entry in data_cfg.get("mix_paths", []):
        path = Path(entry["path"])
        if not path.is_absolute() and not path.exists():
            path = data_dir / path
        entries.append(
            {
                "path": str(path),
                "name": entry.get("name", "unknown"),
                "weight": float(entry.get("weight", 1.0)),
            }
        )
    return entries


def resolve_eval_path(data_cfg: dict[str, Any], data_dir: Path) -> Path | None:
    configured = data_cfg.get("eval_path")
    if not configured:
        return None
    path = Path(configured)
    if not path.is_absolute():
        path = data_dir / path
    if not path.exists():
        raise FileNotFoundError(f"eval path not found: {path}")
    return path


def build_full_dataloader(
    cfg: dict[str, Any],
    train_path: Path | None,
    data_dir: Path,
    use_prefix_lm: bool,
    device: str,
    eval_samples: int = 0,
    dataset_mode: str = "memmap",
    sequential_cycles: int = 0,
    steps_per_cycle: int | None = None,
) -> tuple[Any, Any | None, int, int, bool]:
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    seq_len = int(data_cfg.get("max_seq_len", 512))
    # Variable-length training: windows sampled in [min_seq_len, seq_len].
    # Defaults to seq_len (fixed length) when unset. Eval always uses seq_len
    # for a deterministic held-out set.
    min_seq_len_cfg = data_cfg.get("min_seq_len")
    min_seq_len = int(min_seq_len_cfg) if min_seq_len_cfg is not None else seq_len
    batch_size = int(train_cfg.get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", device.startswith("cuda")))
    dataset_mode = dataset_mode.lower()
    if dataset_mode == "memmap_packed":
        dataset_mode = "memmap"

    if dataset_mode == "sft":
        tokenizer_name = str(data_cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
        tokenizer = TokenizerWrapper.smollm2(tokenizer_name, use_fast=True)
        dataset_name = str(data_cfg.get("dataset_name", "HuggingFaceTB/smoltalk"))
        local_path = data_cfg.get("local_path")
        train_loader = get_sft_dataloader(
            dataset_name=dataset_name,
            tokenizer=tokenizer.tokenizer,
            max_seq_len=seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            local_path=local_path,
        )
        eval_loader = None
        return train_loader, eval_loader, batch_size, seq_len, pin_memory

    dtype = data_cfg.get("dtype", "uint16")

    def _make_loader(ds: Any, shuffle: bool, drop_last: bool = True) -> Any:
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
        }
        if use_prefix_lm:
            kwargs["collate_fn"] = partial(prefix_lm_collate, seq_len=seq_len)
        else:
            kwargs["collate_fn"] = _shift_collate
        if num_workers > 0:
            kwargs["prefetch_factor"] = 4
            kwargs["persistent_workers"] = True
        return DataLoader(ds, **kwargs)

    mix_paths = resolve_mix_paths(data_cfg, data_dir) if train_path is None else []
    if mix_paths:
        if use_prefix_lm:
            raise ValueError("mix_paths does not support prefix_lm")
        if sequential_cycles > 0:
            entries = _resolve_sequential_entries(data_cfg, data_dir)
            train_loader = SequentialCyclingIterator(
                entries,
                batch_size=batch_size,
                seq_len=seq_len,
                num_workers=num_workers,
                pin_memory=pin_memory,
                dtype=dtype,
                cycles_per_dataset=sequential_cycles,
                steps_per_cycle=steps_per_cycle,
                min_seq_len=min_seq_len,
            )
        else:
            train_loader = get_mixed_memmap_dataloader(
                mix_paths,
                batch_size=batch_size,
                seq_len=seq_len,
                num_workers=num_workers,
                pin_memory=pin_memory,
                dtype=dtype,
                seed=int(train_cfg.get("seed", 0)),
                preload=True,
                min_seq_len=min_seq_len,
            )
        eval_path = resolve_eval_path(data_cfg, data_dir)
        eval_loader = (
            _make_loader(
                MemmapDataset(eval_path, seq_len=seq_len, dtype=dtype), shuffle=False
            )
            if eval_path is not None
            else None
        )
        return train_loader, eval_loader, batch_size, seq_len, pin_memory

    if train_path is None:
        raise ValueError("train_path is required when no mix_paths")
    dataset = MemmapDataset(
        train_path, seq_len=seq_len, dtype=dtype, min_seq_len=min_seq_len
    )
    total = len(dataset)
    if total <= 0:
        raise ValueError(f"dataset is empty: {train_path}")
    if eval_samples > 0:
        eval_samples = min(eval_samples, total // 10)
        train_ds = Subset(cast(Any, dataset), list(range(total - eval_samples)))
        eval_ds = Subset(cast(Any, dataset), list(range(total - eval_samples, total)))
    else:
        train_ds = dataset
        eval_ds = None
    if len(train_ds) < batch_size:
        raise ValueError(
            f"train dataset size {len(train_ds)} < batch_size {batch_size}"
        )

    train_loader = _make_loader(train_ds, shuffle=True, drop_last=True)
    eval_loader = (
        _make_loader(eval_ds, shuffle=False, drop_last=False)
        if eval_ds is not None
        else None
    )
    return train_loader, eval_loader, batch_size, seq_len, pin_memory


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    eval_loader: Any,
    device: str,
    pin_memory: bool,
    precision: str,
) -> dict[str, float]:
    """In-loop eval: perplexity + accuracy over the eval dataloader."""
    was_training = model.training
    model.eval()
    total_ce = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0, device=device)
    total_correct = torch.tensor(0, device=device)
    num_batches = 0
    for batch, targets in eval_loader:
        batch = to_device(batch, device, pin_memory)
        targets = targets.to(device, non_blocking=pin_memory)
        targets = apply_prefix_mask(targets, batch)
        tokens = batch.tokens if isinstance(batch, PrefixLMBatch) else batch
        with autocast_ctx(precision, device):
            output = model(tokens, training_mode=False)
            logits = output["logits"] if isinstance(output, dict) else output[0]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_ce += ce
            valid = targets != -100
            total_tokens += valid.sum()
            preds = logits.argmax(dim=-1)
            total_correct += ((preds == targets) & valid).sum()
        num_batches += 1
    if was_training:
        model.train()
    if num_batches == 0:
        return {"ppl": float("nan"), "acc": float("nan")}
    total_tokens = int(total_tokens.item())
    return {
        "ppl": math.exp(float(total_ce.item()) / max(1, total_tokens)),
        "acc": 100.0 * float(total_correct.item()) / max(1, total_tokens),
    }


def print_model_summary(model: HAGI, cfg: Any, device: str) -> None:
    params = (
        model.num_parameters()
        if hasattr(model, "num_parameters")
        else sum(p.numel() for p in model.parameters())
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        vram_text = f"{reserved:.2f}GB reserved / {vram:.2f}GB total"
    else:
        vram_text = "n/a"
    print(
        f"model summary | params {params:,} | vram {vram_text} | "
        f"use_loop {cfg.use_loop} | hrm {cfg.hrm} | use_gdr {cfg.use_gdr} | "
        f"hdim_full {cfg.hdim_full}"
    )


def run_dry_profile(
    model: torch.nn.Module,
    dataloader: Any,
    device: str,
    use_prefix_lm: bool,
    precision: str,
) -> None:
    model.eval()
    batch = next(iter(dataloader))
    batch = to_device(batch, device, non_blocking=device.startswith("cuda"))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), autocast_ctx(precision, device):
        if use_prefix_lm:
            inputs, targets = batch
            output = model(inputs.tokens)
            logits = output["logits"] if isinstance(output, dict) else output
        else:
            x, targets = batch
            output = model(x, targets=targets)
            logits = output["logits"] if isinstance(output, dict) else output[0]
        loss = output.get("loss") if isinstance(output, dict) else None
        if loss is None and logits is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        assert isinstance(loss, torch.Tensor), "dry-run produced no loss"
    print(f"dry_run_loss {float(loss.detach().cpu()):.4f}")
    if device.startswith("cuda"):
        allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
        reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
        print(
            f"dry_run_peak_vram allocated {allocated:.2f}GB | reserved {reserved:.2f}GB"
        )
    print("dry_run_complete no optimizer step executed")


def _build_loop_config(
    cfg: dict[str, Any], ckpt_dir: Path, max_steps: int
) -> LoopConfig:
    """Delegates to the canonical builder in hagi.train.loop (shared with cli)."""
    from hagi.train.loop import build_loop_config

    return build_loop_config(cfg, str(ckpt_dir), max_steps)


def run(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Single canonical training entry: build model/optimizer/data/loop -> train."""
    model_cfg = config_from_dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("training", {}))
    data_cfg = cfg.get("data", {})
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate

    # Resolve resume BEFORE model construction so the checkpoint's config wins.
    start_step = 0
    if args.resume is not None and args.resume.exists():
        from hagi.train.loop import load_checkpoint

        model, start_step, _ = load_checkpoint(str(args.resume), args.device)
        model_cfg = model.cfg
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        model = HAGI(model_cfg).to(args.device)

    if hasattr(model.cfg, "gradient_checkpointing"):
        model.cfg.gradient_checkpointing = bool(
            train_cfg.get("gradient_checkpointing", model.cfg.gradient_checkpointing)
        )

    use_prefix_lm = bool(train_cfg.get("use_prefix_lm", False))
    dataset_mode = getattr(args, "dataset_mode", None) or str(
        data_cfg.get("dataset_mode", "memmap")
    )
    train_path = (
        resolve_train_path(cfg, args.train_path, args.data_dir)
        if args.train_path is not None or not resolve_mix_paths(data_cfg, args.data_dir)
        else None
    )
    eval_interval = int(train_cfg.get("eval_interval", 0))
    eval_samples = int(train_cfg.get("eval_samples", 500))
    dataset_cycles = getattr(args, "dataset_cycles", None)
    sequential_cycles = int(
        dataset_cycles
        if dataset_cycles is not None
        else data_cfg.get("sequential_cycles", 0)
    )

    _seq_len = int(data_cfg.get("max_seq_len", 512))
    _batch_size = int(train_cfg.get("batch_size", 2))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 4))
    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be > 0, got {grad_accum_steps}")
    max_steps = (
        int(args.max_steps)
        if args.max_steps is not None
        else _resolve_max_steps(train_cfg, data_cfg, _batch_size, _seq_len)
    )
    steps_per_cycle: int | None = None
    if sequential_cycles > 0:
        num_datasets = len(data_cfg.get("mix_paths", []))
        if num_datasets > 0:
            total_batches = max_steps * grad_accum_steps
            steps_per_cycle = max(
                1, total_batches // (num_datasets * sequential_cycles)
            )

    dataloader, eval_loader, batch_size, seq_len, pin_memory = build_full_dataloader(
        cfg,
        train_path,
        args.data_dir,
        use_prefix_lm,
        args.device,
        eval_samples=eval_samples if eval_interval > 0 else 0,
        dataset_mode=dataset_mode,
        sequential_cycles=sequential_cycles,
        steps_per_cycle=steps_per_cycle,
    )

    print_model_summary(model, model_cfg, args.device)
    if args.dry_run:
        run_dry_profile(
            model,
            dataloader,
            args.device,
            use_prefix_lm,
            train_cfg.get("precision", "bf16"),
        )
        return

    optimizer = build_optimizer(model, train_cfg)
    resumed_ema_state: dict[str, Any] | None = None
    if args.resume is not None and args.resume.exists():
        if args.resume.is_dir():
            optimizer_path = args.resume / "optimizer.pt"
            if optimizer_path.exists():
                try:
                    optimizer.load_state_dict(
                        torch.load(
                            optimizer_path,
                            map_location="cpu",
                            weights_only=True,
                        )
                    )
                    print("loaded optimizer state")
                except Exception as exc:
                    print(f"optimizer state mismatch, starting fresh: {exc}")
            ema_path = args.resume / "ema.pt"
            if ema_path.exists():
                try:
                    ema_state = torch.load(
                        ema_path, map_location="cpu", weights_only=True
                    )
                    for key in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
                        ema_state.pop(key, None)
                    resumed_ema_state = ema_state
                    print("found EMA state to restore")
                except Exception as exc:
                    print(f"EMA state unreadable, starting fresh: {exc}")
        else:
            state = torch.load(args.resume, map_location="cpu", weights_only=True)
            if "optimizer" in state:
                try:
                    optimizer.load_state_dict(state["optimizer"])
                    print("loaded optimizer state")
                except Exception as exc:
                    print(f"optimizer state mismatch, starting fresh: {exc}")
            if "model_ema" in state:
                ema_state = dict(state["model_ema"])
                for key in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
                    ema_state.pop(key, None)
                resumed_ema_state = ema_state
                print("found EMA state to restore")
            del state  # free checkpoint tensors from CPU RAM

    # Move optimizer state to the target device (loaded from CPU to save
    # VRAM during resume) and release CUDA cache accumulated during loading.
    if args.resume is not None and args.resume.exists():
        for _st in optimizer.state.values():
            for _k, _v in _st.items():
                if isinstance(_v, torch.Tensor):
                    _st[_k] = _v.to(args.device)
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    loop_cfg = _build_loop_config(cfg, args.ckpt_dir, max_steps)

    maybe_disable_gradient_checkpointing(
        model,
        args.device,
        batch_size=max(1, int(train_cfg.get("batch_size", 8)) // 2),
        seq_len=seq_len,
        target_batch=int(train_cfg.get("batch_size", 8)),
        safety=float(train_cfg.get("vram_probe", {}).get("safety", 1.0)),
        headroom=float(train_cfg.get("vram_probe", {}).get("headroom", 0.85)),
        explicit="gradient_checkpointing" in cfg.get("training", {}),
    )
    print_model_size(model)
    print_vram_usage()

    # Restore EMA weights from the resumed checkpoint so accumulation continues
    # from the saved EMA model rather than the raw main weights. When EMA is
    # enabled but no state is present, train() builds a fresh copy.
    import copy

    model_ema: torch.nn.Module | None = None
    if loop_cfg.enable_ema:
        model_ema = copy.deepcopy(model)
        model_ema.eval()
        for param in model_ema.parameters():
            param.requires_grad_(False)
        if resumed_ema_state is not None:
            try:
                model_ema.load_state_dict(resumed_ema_state, strict=False)
                print("loaded EMA state")
            except Exception as exc:
                print(f"EMA state mismatch, EMA starts from main weights: {exc}")

    # Full in-loop eval (ppl/acc) over the held-out loader every eval_interval.
    def _on_eval(step: int) -> None:
        if eval_loader is None:
            return
        eval_model = (
            model_ema
            if (step >= loop_cfg.ema_start_step and model_ema is not None)
            else model
        )
        metrics = run_eval(
            eval_model,
            eval_loader,
            args.device,
            pin_memory,
            train_cfg.get("precision", "bf16"),
        )
        eval_tag = "ema" if step >= loop_cfg.ema_start_step else "model"
        print(
            f"eval | step {step:6d} | ppl {metrics['ppl']:.2f} | "
            f"acc {metrics['acc']:.1f}% | {eval_tag}"
        )

    train(
        model,
        optimizer,
        dataloader,
        loop_cfg,
        device=args.device,
        start_step=start_step,
        on_eval=_on_eval if (eval_interval > 0 and eval_loader is not None) else None,
        model_ema=model_ema,
        batched=True,
        use_prefix_lm=use_prefix_lm,
        to_device_fn=to_device,
        apply_prefix_mask_fn=apply_prefix_mask,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build model and one batch, report memory, then exit before optimizer step",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["memmap", "memmap_packed", "sft"],
        default=None,
        help="dataset loading mode (overrides config)",
    )
    parser.add_argument(
        "--dataset-cycles",
        type=int,
        default=None,
        help="cycles per dataset in sequential mode (overrides config)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="logging interval in steps (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.log_interval is not None:
        cfg.setdefault("training", {})["log_interval"] = args.log_interval
    run(args, cfg)


if __name__ == "__main__":
    main()
