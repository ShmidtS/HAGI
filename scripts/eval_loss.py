"""Compare ablation checkpoints by validation loss on a fixed batch set.

Every checkpoint is scored on the **same** K random batches (one fixed seed builds
the batches once, reused for all models), so differences in loss/perplexity reflect
the model, not the data. The decisive comparison is B vs D — same params, same
batches, the only difference is grade decomposition.

    python scripts/eval_loss.py --data /content/drive/MyDrive/hagi-data \
        --ckpt checkpoints/ablation_{a,b,c,d}/step-00007630.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from prototype.data.dataset import MemmapTokenDataset
from prototype.training.loop import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description="Score checkpoints on a shared fixed batch set.")
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more step-*.pt paths")
    ap.add_argument("--data", required=True, help="tokenized .bin shard dir (held-out or train)")
    ap.add_argument("--batches", type=int, default=50, help="number of batches to average over")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1234, help="fixes the eval batch set for all models")
    args = ap.parse_args()

    # Build the eval batches ONCE so every checkpoint sees identical data.
    ds = MemmapTokenDataset(args.data)
    gen = np.random.default_rng(args.seed)
    batches = [
        ds.get_batch(args.batch_size, args.seq, device=args.device, generator=gen)
        for _ in range(args.batches)
    ]
    tokens = args.batches * args.batch_size * args.seq
    print(f"eval set: {args.batches} batches x {args.batch_size} x {args.seq} = {tokens:,} tokens "
          f"(seed {args.seed})\n")

    print(f"{'checkpoint':<52} {'step':>7} {'loss':>8} {'ppl':>9}")
    rows = []
    for path in args.ckpt:
        model, step = load_checkpoint(path, device=args.device)
        model.eval()
        total = 0.0
        with torch.no_grad():
            for x, y in batches:
                _, loss = model(x, targets=y)
                total += loss.item()
        avg = total / len(batches)
        ppl = math.exp(avg)
        rows.append((os.path.basename(os.path.dirname(path)) or path, avg, ppl))
        print(f"{path:<52} {step:>7} {avg:>8.4f} {ppl:>9.2f}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Decisive read: B vs D (grade decomposition isolated), C vs D (integrated vs bolted-on).
    by = {name: (loss, ppl) for name, loss, ppl in rows}
    if "ablation_b" in by and "ablation_d" in by:
        db = by["ablation_d"][0] - by["ablation_b"][0]
        verdict = "D better" if db < 0 else "B better" if db > 0 else "tie"
        print(f"\nB vs D: loss D-B = {db:+.4f}  ({verdict}; negative = grade decomposition helps)")
    if "ablation_c" in by and "ablation_d" in by:
        dc = by["ablation_d"][0] - by["ablation_c"][0]
        print(f"C vs D: loss D-C = {dc:+.4f}  (negative = integrated GDR beats bolted-on Clifford)")


if __name__ == "__main__":
    main()
