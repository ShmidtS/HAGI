"""Compare EMA vs raw model weights on perplexity and token entropy.

Usage:
    python scripts/compare_ema.py --checkpoint checkpoints/rtx3070/step-00042000.pt
    python scripts/compare_ema.py --checkpoint checkpoints/rtx3070/step-00042000.pt --data data/v4_3b/tinystories.bin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from hagi.model import HAGI
from hagi.train.config import config_from_dict
from hagi.train.loop import load_checkpoint, _convert_split_qkv_to_fused


@torch.no_grad()
def evaluate_model(
    model: HAGI,
    data: torch.Tensor,
    seq_len: int,
    n_batches: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_entropy = 0.0

    # Temporarily disable fused CE so we get logits for entropy computation.
    original_fused_ce = model.cfg.use_fused_ce
    model.cfg.use_fused_ce = False

    for i in range(n_batches):
        start = (i * seq_len) % (len(data) - seq_len - 1)
        x = data[start : start + seq_len].unsqueeze(0).to(device)
        y = data[start + 1 : start + 1 + seq_len].unsqueeze(0).to(device)

        output = model(x, targets=y)
        if isinstance(output, dict):
            loss = output["loss"]
            logits = output.get("logits")
        elif isinstance(output, tuple):
            logits, loss = output[0], output[1]
        else:
            loss = output
            logits = None

        total_loss += loss.item() * seq_len
        total_tokens += seq_len

        if logits is not None:
            logits_flat = logits.reshape(-1, logits.size(-1))
            log_probs = F.log_softmax(logits_flat, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)
            total_entropy += entropy.sum().item()

    model.cfg.use_fused_ce = original_fused_ce

    avg_loss = total_loss / total_tokens
    avg_entropy = total_entropy / total_tokens if total_entropy > 0 else float("nan")
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return {
        "loss": avg_loss,
        "ppl": ppl,
        "entropy": avg_entropy,
    }


def load_raw_weights(checkpoint_path: str, device: str) -> tuple[HAGI, int]:
    """Load model with RAW (non-EMA) weights from a flat checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = config_from_dict(state["config"])
    model = HAGI(cfg)
    state_dict = state["model"]
    state_dict = _convert_split_qkv_to_fused(model, state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, int(state.get("step", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EMA vs raw weights")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Flat .pt checkpoint"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/v4_3b/tinystories.bin"),
        help="memmap .bin data file",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-batches", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    seq_len = args.seq_len
    n_batches = args.n_batches

    # Load eval data
    data = torch.frombuffer(args.data.read_bytes(), dtype=torch.uint16).long()
    print(f"Data: {args.data} ({len(data)} tokens)")

    # --- Raw weights ---
    print(f"\nLoading RAW weights from {args.checkpoint}...")
    model_raw, step = load_raw_weights(str(args.checkpoint), device)
    print(f"  step={step}")
    raw_metrics = evaluate_model(model_raw, data, seq_len, n_batches, device)
    del model_raw
    torch.cuda.empty_cache() if device.startswith("cuda") else None

    # --- EMA weights ---
    print(f"\nLoading EMA weights from {args.checkpoint}...")
    model_ema, _, _ = load_checkpoint(str(args.checkpoint), device=device, use_ema=True)
    ema_metrics = evaluate_model(model_ema, data, seq_len, n_batches, device)
    del model_ema
    torch.cuda.empty_cache() if device.startswith("cuda") else None

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"{'Metric':<15} {'Raw':>12} {'EMA':>12} {'Delta':>12} {'Better':>8}")
    print(f"{'='*60}")
    for key in ["loss", "ppl", "entropy"]:
        r = raw_metrics[key]
        e = ema_metrics[key]
        d = e - r
        if key == "entropy":
            better = "EMA" if d > 0 else "Raw"
        else:
            better = "EMA" if d < 0 else "Raw"
        if key == "ppl":
            print(f"{key:<15} {r:>12.2f} {e:>12.2f} {d:>+12.2f} {better:>8}")
        else:
            print(f"{key:<15} {r:>12.4f} {e:>12.4f} {d:>+12.4f} {better:>8}")
    print(f"{'='*60}")
    print("\nLower loss/ppl = better. Higher entropy = more diverse predictions.")


if __name__ == "__main__":
    main()
