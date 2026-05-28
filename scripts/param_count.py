"""Report parameter count and effective depth for a config.

Usage:
    python scripts/param_count.py --config configs/gdr.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prototype.model.hagi import HAGI
from prototype.training.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = HAGI(cfg["model"])
    m = cfg["model"]

    total = sum(p.numel() for p in model.parameters())
    loops = m.loop_count if m.use_loop else 1
    eff_depth = m.perception_layers + m.reasoning_layers * loops + m.expression_layers

    print(f"config            : {cfg['name']}")
    print(f"unique parameters : {total / 1e6:.1f}M")
    print(f"loop count        : {loops}")
    print(f"physical layers   : {m.perception_layers + m.reasoning_layers + m.expression_layers}")
    print(f"effective depth   : {eff_depth} layers")
    print(f"use_loop / use_gdr: {m.use_loop} / {m.use_gdr}")


if __name__ == "__main__":
    main()
