"""Config-level smoke tests.

Every shipped YAML config builds a model, runs a forward pass, produces the
right logits shape and a finite loss, and lands in the expected param band.
Catches config drift (head_dim, grade sums, vocab) without a training run.
"""

import math
from pathlib import Path

import pytest
import torch

from hagi.model import HAGI
from hagi.train.config import config_from_dict, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SHIPPED = ["baseline.yaml", "gdr.yaml", "rtx3070_canonical.yaml"]


@pytest.mark.parametrize("config_name", SHIPPED)
def test_shipped_config_builds_and_runs(config_name):
    cfg = load_config(CONFIG_DIR / config_name)
    model_cfg = config_from_dict(cfg["model"])
    model = HAGI(model_cfg).eval()

    params = model.num_parameters() / 1e6
    assert 1.0 < params < 500.0, (
        f"{config_name}: {params:.1f}M out of expected 1M–500M band"
    )

    B, T = 2, 16
    x = torch.randint(0, model_cfg.vocab_size, (B, T))
    y = torch.randint(0, model_cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits, loss = model(x, targets=y)

    assert logits.shape == (B, T, model_cfg.vocab_size)
    assert math.isfinite(loss.item())
