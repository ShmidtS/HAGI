from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FeedbackEntry:
    prompt_ids: list[int]
    response_ids: list[int]
    reward: float


class FeedbackBuffer:
    def __init__(self, max_size: int = 256) -> None:
        self.max_size = max_size
        self.entries: list[FeedbackEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add(
        self, prompt_ids: list[int], response_ids: list[int], reward: float
    ) -> None:
        self.entries.append(FeedbackEntry(prompt_ids, response_ids, reward))
        if len(self.entries) > self.max_size:
            self.entries.pop(0)

    def sample_batch(
        self, batch_size: int, device: str | torch.device = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.entries:
            return None
        pos = [e for e in self.entries if e.reward > 0]
        neg = [e for e in self.entries if e.reward < 0]
        chosen: list[FeedbackEntry] = []
        if pos and neg:
            half = max(1, batch_size // 2)
            chosen = (pos[:half] + neg[:half])[:batch_size]
        else:
            chosen = self.entries[:batch_size]
        if not chosen:
            return None
        max_len = max(len(e.prompt_ids) + len(e.response_ids) for e in chosen)
        xs = torch.full((len(chosen), max_len), 0, dtype=torch.long, device=device)
        ys = torch.full((len(chosen), max_len), -100, dtype=torch.long, device=device)
        masks = torch.zeros((len(chosen), max_len), dtype=torch.bool, device=device)
        rewards = torch.zeros(len(chosen), dtype=torch.float, device=device)
        for i, e in enumerate(chosen):
            seq = e.prompt_ids + e.response_ids
            xs[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            resp_start = len(e.prompt_ids)
            ys[i, resp_start : len(seq) - 1] = torch.tensor(
                e.response_ids[1:], dtype=torch.long, device=device
            )
            masks[i, resp_start : len(seq) - 1] = True
            rewards[i] = e.reward
        return xs, ys, masks, rewards


class OnlineLearner:
    """Online RL learner for LoRA-style adapters.

    ``adapter`` can be any ``nn.Module`` containing LoRA parameters (e.g. a
    single ``LoRAAdapter`` or an ``nn.ModuleList`` of them). Examples with
    zero reward are treated as negative and push the signed loss toward
    maximizing cross-entropy. Per-example loss is not clamped; the global
    ``grad_clip`` keeps the update stable instead.
    """

    def __init__(
        self,
        adapter: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
    ) -> None:
        self.adapter = adapter
        params = [p for p in adapter.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip

    def _device(self) -> torch.device:
        try:
            return next(self.adapter.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def learn_step(
        self,
        buffer: FeedbackBuffer,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int = 4,
    ) -> float | None:
        batch = buffer.sample_batch(batch_size, device=self._device())
        if batch is None:
            return None
        xs, ys, masks, rewards = batch
        if xs.shape[0] == 0:
            return None
        if masks.sum() == 0:
            return None
        try:
            self.optimizer.zero_grad()
            logits = forward_fn(xs)  # [B, T, V]
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                ys.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(xs.shape[0], xs.shape[1])
            masked_ce = (ce * masks.float()).sum(dim=1) / masks.float().sum(
                dim=1
            ).clamp_min(1.0)
            signed_loss = torch.where(rewards > 0, masked_ce, -masked_ce)
            loss = signed_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.grad_clip)
            self.optimizer.step()
            return loss.item()
        except torch.cuda.OutOfMemoryError:
            self.optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter": self.adapter.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.adapter.load_state_dict(state["adapter"])
        self.optimizer.load_state_dict(state["optimizer"])
