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
from prototype.model.hagi import HAGI, HAGIConfig, cross_entropy_loss
from prototype.model.transformer import TransformerConfig
from prototype.training.config import config_from_dict, config_to_dict, load_config
from prototype.training.loop import (
    latest_checkpoint,
    load_checkpoint,
    resume_into,
    save_checkpoint,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
SHIPPED = ["baseline.yaml", "gdr.yaml", "colab_t4.yaml"]


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


def test_init_loss_near_uniform():
    """Weight init must keep initial logits near-uniform: CE ~ ln(vocab), not ~10x
    it. Guards the over-scaled tied-embedding init the data dry-run surfaced."""
    torch.manual_seed(0)
    model = _tiny_model().eval()  # vocab 64
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    with torch.no_grad():
        _, loss = model(x, targets=y)
    assert loss.item() < 2.0 * math.log(64), (
        f"init loss {loss.item():.2f} >> ln(64)={math.log(64):.2f} — embedding init too large"
    )


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


def test_chunked_cross_entropy_matches_plain():
    """Chunked CE must be numerically identical to the unchunked path — it only
    splits the fp32 upcast to avoid the full-logit memory spike."""
    torch.manual_seed(0)
    logits = torch.randn(37, 11)
    targets = torch.randint(0, 11, (37,))
    assert torch.allclose(
        cross_entropy_loss(logits, targets, chunk_size=0),
        cross_entropy_loss(logits, targets, chunk_size=8),
        atol=1e-6,
    )
    # ...and with masked (ignore_index) positions.
    targets[::5] = -100
    assert torch.allclose(
        cross_entropy_loss(logits, targets, ignore_index=-100, chunk_size=0),
        cross_entropy_loss(logits, targets, ignore_index=-100, chunk_size=8),
        atol=1e-6,
    )


def test_gradient_checkpointing_matches_plain():
    """gradient_checkpointing must not change the loss — only how activations are
    stored for backward. Same weights, same inputs, same forward result."""
    torch.manual_seed(0)
    plain = _tiny_model()
    ckpt_model = _tiny_model()
    ckpt_model.load_state_dict(plain.state_dict())
    ckpt_model.cfg.gradient_checkpointing = True

    plain.train()
    ckpt_model.train()
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    _, l_plain = plain(x, targets=y)
    _, l_ckpt = ckpt_model(x, targets=y)
    assert torch.allclose(l_plain, l_ckpt, atol=1e-5)

    l_ckpt.backward()  # backward through checkpointed blocks must produce finite grads
    grads = [p.grad for p in ckpt_model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_resume_restores_model_and_optimizer(tmp_path):
    """resume_into restores weights AND optimizer state — the contract that lets a
    killed 12h free-tier session continue exactly where it stopped."""
    torch.manual_seed(0)
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    _, loss = model(x, targets=y)
    loss.backward()
    opt.step()
    save_checkpoint(model, opt, step=7, ckpt_dir=str(tmp_path))

    torch.manual_seed(123)  # different init — resume must overwrite it
    model2 = _tiny_model()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    path = latest_checkpoint(str(tmp_path))
    assert path is not None
    step = resume_into(model2, opt2, str(path))

    assert step == 7
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.allclose(p1, p2, atol=1e-6)
    assert len(opt2.state) > 0  # AdamW moment buffers restored
