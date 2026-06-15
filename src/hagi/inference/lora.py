from __future__ import annotations

import torch
from torch import nn


class LoRAAdapter(nn.Module):
    """Low-rank adapter wrapped around a linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(
            torch.randn(rank, base.in_features, device=dev, dtype=dt) * 0.01
        )
        self.B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=dev, dtype=dt)
        )
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)


def apply_lora_to_model(
    model: nn.Module,
    target_patterns: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    rank: int = 8,
    alpha: int = 16,
) -> tuple[nn.ModuleList, list[str]]:
    """Wrap matching nn.Linear layers in-place and return the combined adapter + names.

    Idempotent: re-applying to an already-wrapped model reuses the existing
    adapters instead of wrapping their internal base layers again.
    """
    adapters: list[LoRAAdapter] = []
    names: list[str] = []
    adapter_names: set[str] = set()
    for name, module in list(model.named_modules()):
        # Skip modules already wrapped, and anything nested inside an adapter
        # (e.g. its base Linear) so a second call does not double-wrap.
        if any(name == p or name.startswith(p + ".") for p in adapter_names):
            continue
        if isinstance(module, LoRAAdapter):
            adapter_names.add(name)
            if any(p.lower() in name.lower() for p in target_patterns):
                adapters.append(module)
                names.append(name)
            continue
        if isinstance(module, nn.Linear) and any(
            p.lower() in name.lower() for p in target_patterns
        ):
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if parent_name == "" else model.get_submodule(parent_name)
            adapter = LoRAAdapter(module, rank=rank, alpha=alpha)
            setattr(parent, attr_name, adapter)
            adapter_names.add(name)
            adapters.append(adapter)
            names.append(name)
    if not adapters:
        raise ValueError(f"No nn.Linear modules matched patterns {target_patterns}")
    return nn.ModuleList(adapters), names
