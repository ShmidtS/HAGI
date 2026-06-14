"""Compression observability metrics for HAGI.

Stateless functions + a thin CompressionMonitor orchestrator. Metrics take
tensors already materialized in the forward pass (logits, hidden, targets) and
NEVER enter the loss / backward graph. The monitor is not part of the model
state_dict — checkpoint compatibility is unaffected.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def entropy_rate(logits: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Natural entropy H[p] = -sum p log p over the last dim.

    Args:
        logits: [..., V] raw logits.
        eps: numerical floor for log.

    Returns:
        [...]-shaped per-position entropy in nats.
    """
    probs = F.softmax(logits.float(), dim=-1)
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def compression_ratio(
    num_params: int, train_tokens: int, bytes_per_param: int = 2
) -> float:
    """Degree of compression: train_bytes / model_bytes.

    Args:
        num_params: total trainable parameter count.
        train_tokens: total tokens in the training corpus.
        bytes_per_param: storage width of a parameter (2 = fp16/bf16).

    Returns:
        Scalar ratio >= 0. Higher = more compressed.
    """
    model_bytes = num_params * bytes_per_param
    if model_bytes <= 0:
        return 0.0
    # tokens stored as int32 token ids (4 bytes) — but the canonical framing
    # compares param-bytes vs token-bytes; use same unit on both sides.
    train_bytes = train_tokens * bytes_per_param
    return float(train_bytes / model_bytes)


def calibration_error(
    confidence: torch.Tensor,
    correctness: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE): mean |acc - conf| over confidence bins.

    Args:
        confidence: [N] max-probability per prediction in [0, 1].
        correctness: [N] binary (1 = argmax == target).
        n_bins: number of equal-width confidence bins.

    Returns:
        Scalar ECE in [0, 1]. Returns 0.0 for empty input.
    """
    if confidence.numel() == 0:
        return 0.0
    conf = confidence.float().reshape(-1)
    corr = correctness.float().reshape(-1)
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=conf.device)
    ece = torch.zeros((), device=conf.device)
    total = conf.numel()
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum().item())
        if count == 0:
            continue
        bin_acc = corr[mask].mean()
        bin_conf = conf[mask].mean()
        ece = ece + (count / total) * (bin_acc - bin_conf).abs()
    return float(ece.item())


def effective_rank(hidden: torch.Tensor) -> float:
    """Information-density proxy via stable log singular values.

    Uses exp(entropy(singular_value_distribution)), i.e. the "effective rank"
    of Roy & Vetterli. Cheaper and more stable than full PCA on the hidden
    covariance. Input is flattened to [N, D] first.

    Args:
        hidden: [..., D] hidden states.

    Returns:
        Scalar effective rank >= 0. Returns 0.0 for degenerate input.
    """
    flat = hidden.float().reshape(-1, hidden.shape[-1])
    if flat.shape[0] < 2:
        return 0.0
    try:
        s = torch.linalg.svdvals(flat)
    except RuntimeError:
        return 0.0
    s = s.clamp_min(1e-12)
    p = s / s.sum()
    p = p.clamp_min(1e-12)
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


def artifact_ratio(confidence: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of low-confidence positions (likely compression artifacts).

    Args:
        confidence: [...] max-probability per position in [0, 1].
        threshold: positions below this count as artifacts.

    Returns:
        Scalar ratio in [0, 1]. Returns 0.0 for empty input.
    """
    conf = confidence.float().reshape(-1)
    if conf.numel() == 0:
        return 0.0
    return float((conf < threshold).float().mean().item())


class CompressionMonitor:
    """Orchestrates compression metrics with correct frequency.

    Heavy scalar (compression_ratio) is computed once at construction. Light
    metrics (entropy, calibration, artifact_ratio, confidence) run on every
    compute() call; effective_rank runs every heavy_interval_mult steps on a
    token subsample. The monitor holds no model parameters and is never part
    of a checkpoint state_dict.

    Metrics are OBSERVATION-ONLY: they must never enter the loss / backward
    graph (Goodhart's Law).
    """

    def __init__(
        self,
        num_params: int,
        train_tokens: int,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        cfg = cfg or {}
        self.artifact_threshold = float(cfg.get("artifact_threshold", 0.5))
        self.calibration_bins = int(cfg.get("calibration_bins", 10))
        self.rank_subsample = int(cfg.get("rank_subsample", 256))
        self.heavy_interval_mult = int(cfg.get("heavy_interval_mult", 4))
        self._compression_ratio = compression_ratio(num_params, train_tokens)

    def compute(
        self,
        logits: torch.Tensor | None,
        hidden: torch.Tensor | None,
        targets: torch.Tensor | None,
        step: int,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {
            "compression_ratio": self._compression_ratio,
        }
        if logits is None or targets is None:
            return metrics

        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            confidence = probs.max(dim=-1).values  # [B, T]
            preds = logits.argmax(dim=-1)  # [B, T]
            mask = targets.reshape(-1) != -100

            ent = entropy_rate(logits)
            metrics["entropy"] = float(ent.reshape(-1)[mask].mean().item())

            conf_flat = (
                confidence.reshape(-1)[mask] if mask.any() else confidence.reshape(-1)
            )
            metrics["artifact_ratio"] = artifact_ratio(
                conf_flat, threshold=self.artifact_threshold
            )

            if mask.any():
                corr = (preds.reshape(-1)[mask] == targets.reshape(-1)[mask]).float()
                metrics["calibration_error"] = calibration_error(
                    conf_flat, corr, n_bins=self.calibration_bins
                )
                metrics["accuracy"] = float(corr.mean().item())
            metrics["avg_confidence"] = float(conf_flat.mean().item())

            if hidden is not None and (step % max(1, self.heavy_interval_mult) == 0):
                flat_hidden = hidden.float().reshape(-1, hidden.shape[-1])
                n = flat_hidden.shape[0]
                if n > self.rank_subsample:
                    idx = torch.randperm(n, device=flat_hidden.device)[
                        : self.rank_subsample
                    ]
                    flat_hidden = flat_hidden[idx]
                metrics["effective_rank"] = effective_rank(flat_hidden)

        return metrics
