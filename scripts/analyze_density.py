"""Neuron-density diagnostic: find underutilized / redundant parameters.

Loads a trained checkpoint and reports per-submodule:
  - weight magnitude stats (mean abs, max)
  - effective rank (via singular values) of 2D weights
  - dead-near-zero fraction (params below threshold)
  - redundancy (low singular-value ratio => over-parameterized)

Identifies where params are dense vs wasted so architecture can be reshaped
for quality at equal-or-lower compute.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def weight_stats(name: str, t: torch.Tensor) -> dict:
    t2 = t.detach().float()
    abs_t = t2.abs()
    near_zero = (abs_t < 1e-3).float().mean().item()
    stats = {
        "name": name,
        "shape": tuple(t.shape),
        "n": t2.numel(),
        "mean_abs": abs_t.mean().item(),
        "max_abs": abs_t.max().item(),
        "std": t2.std().item(),
        "near_zero_frac": near_zero,
    }
    if t2.ndim == 2:
        try:
            s = torch.linalg.svdvals(t2).clamp_min(1e-12)
            s_norm = s / s.sum()
            s_norm = s_norm.clamp_min(1e-12)
            eff_rank = float(torch.exp(-(s_norm * s_norm.log()).sum()).item())
            stats["eff_rank"] = eff_rank
            stats["min_dim"] = min(t2.shape)
            stats["density"] = eff_rank / min(t2.shape)
            # tail energy: fraction of spectral energy in bottom half of singular values
            half = len(s) // 2
            stats["tail_energy"] = float((s[half:] ** 2).sum().item() / (s ** 2).sum().item())
        except Exception:
            pass
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/rtx3070/step-00102000/model.pt"))
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # Normalize compiled prefixes
    state = {(k.replace("hrm._orig_mod.", "hrm.", 1) if k.startswith("hrm._orig_mod.") else k): v for k, v in state.items()}

    rows = []
    total_n = 0
    total_near = 0
    for name, t in state.items():
        if not t.is_floating_point():
            continue
        rows.append(weight_stats(name, t))
        total_n += t.numel()
        total_near += int((t.detach().float().abs() < 1e-3).sum().item())

    # Group by submodule prefix
    from collections import defaultdict

    def _new_group() -> dict[str, Any]:
        return {"n": 0, "near": 0, "params": []}

    groups: dict[str, dict[str, Any]] = defaultdict(_new_group)
    for r in rows:
        prefix = r["name"].split(".")[0]
        groups[prefix]["n"] += r["n"]
        groups[prefix]["near"] += int(r["near_zero_frac"] * r["n"])
        if "eff_rank" in r:
            groups[prefix]["params"].append(r)

    print(f"=== SUBMODULE PARAM UTILIZATION (ckpt {args.ckpt.name}) ===")
    print(f"{'submodule':14s} {'params':>12s} {'near_zero%':>11s} {'low-density 2D params':>22s}")
    for prefix in sorted(groups, key=lambda k: -groups[k]["n"]):
        g = groups[prefix]
        nz = g["near"] / g["n"] * 100 if g["n"] else 0
        low_den = sum(1 for r in g["params"] if r.get("density", 1) < 0.5)
        ratio = f"{low_den}/{len(g['params'])}"
        print(f"{prefix:14s} {g['n']:>12,} {nz:>10.1f}% {ratio:>22s}")

    print(f"\n{'TOTAL params':14s} {total_n:>12,}  near-zero {total_near/total_n*100:.1f}%")

    print(f"\n=== LOWEST-DENSITY 2D WEIGHTS (eff_rank/min_dim < 0.6 => redundant capacity) ===")
    dens = sorted([r for r in rows if "density" in r], key=lambda r: r["density"])
    print(f"{'param':50s} {'shape':16s} {'eff_rank':>9s} {'density':>8s} {'tail_E':>8s}")
    for r in dens[:25]:
        print(f"{r['name'][:50]:50s} {str(r['shape']):16s} {r.get('eff_rank',0):>9.1f} {r.get('density',0):>8.2f} {r.get('tail_energy',0):>8.2%}")

    print(f"\n=== HIGHEST near_zero_frac (dead/redundant weights) ===")
    dead = sorted(rows, key=lambda r: -r["near_zero_frac"])
    print(f"{'param':50s} {'near_zero':>10s} {'mean_abs':>9s} {'std':>8s}")
    for r in dead[:15]:
        print(f"{r['name'][:50]:50s} {r['near_zero_frac']:>9.1%} {r['mean_abs']:>9.4f} {r['std']:>8.4f}")


if __name__ == "__main__":
    main()
