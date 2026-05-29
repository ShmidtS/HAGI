"""Config-level smoke tests + checkpoint roundtrip.

Cheap correctness gates that run on CPU before any GPU training:

  - Every shipped YAML config builds a model, runs a forward pass, produces the
    right logits shape and a finite loss, and lands in the expected param band.
    Catches config drift (head_dim, grade sums, vocab) without a training run.
  - save_checkpoint -> load_checkpoint roundtrips exactly and loads under torch's
    default weights_only=True (regression guard for the pickled-dataclass bug).
"""

import math
from pathlib import Path

import pytest
import torch

from prototype.model.gdr import GradeConfig
from prototype.model.hagi import HAGI, HAGIConfig
from prototype.model.transformer import TransformerConfig
from prototype.training.config import config_from_dict, config_to_dict, load_config
from prototype.training.loop import load_checkpoint, save_checkpoint

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
SHIPPED = ["baseline.yaml", "gdr.yaml"]


def _tiny_model() -> HAGI:
    tcfg = TransformerConfig(hidden_size=64, num_query_heads=4, num_kv_heads=2,
                             intermediate_size=128, max_seq_len=64)
    grades = GradeConfig(scalar=8, vector=16, bivector=16, trivector=8, residual=16)
    cfg = HAGIConfig(vocab_size=64, hidden_size=64,
                     perception_layers=1, reasoning_layers=1, expression_layers=1,
                     loop_count=2, use_loop=True, use_gdr=True,
                     transformer=tcfg, grades=grades)
    return HAGI(cfg)


@pytest.mark.parametrize("config_name", SHIPPED)
def test_shipped_config_builds_and_runs(config_name):
    cfg = load_config(CONFIG_DIR / config_name)
    model = HAGI(cfg["model"]).eval()

    params = model.num_parameters() / 1e6
    assert 100.0 < params < 130.0, f"{config_name}: {params:.1f}M out of expected ~115M band"

    B, T = 2, 16
    x = torch.randint(0, cfg["model"].vocab_size, (B, T))
    y = torch.randint(0, cfg["model"].vocab_size, (B, T))
    with torch.no_grad():
        logits, loss = model(x, targets=y)

    assert logits.shape == (B, T, cfg["model"].vocab_size)
    assert math.isfinite(loss.item())


def test_config_dict_roundtrip():
    cfg = _tiny_model().cfg
    restored = config_from_dict(config_to_dict(cfg))
    assert restored == cfg


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model().eval()
    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        ref = model(x)

    save_checkpoint(model, None, step=42, ckpt_dir=str(tmp_path))
    ckpt = tmp_path / "step-00000042.pt"
    assert ckpt.exists()

    # Loads under default weights_only=True (no pickled dataclass).
    loaded = torch.load(ckpt, weights_only=True)
    assert isinstance(loaded["config"], dict)

    restored, step = load_checkpoint(str(ckpt), device="cpu")
    assert step == 42
    restored.eval()
    with torch.no_grad():
        out = restored(x)
    assert torch.allclose(out, ref, atol=1e-6)
