"""Tests for compression observability metrics."""

import math

import torch

from hagi.train.compression_metrics import (
    CompressionMonitor,
    artifact_ratio,
    calibration_error,
    compression_ratio,
    effective_rank,
    entropy_rate,
)


def test_entropy_rate_uniform_is_max():
    """Uniform distribution over V classes => entropy = log(V) (nats)."""
    V = 49152
    logits = torch.zeros(2, 5, V)  # uniform after softmax
    ent = entropy_rate(logits)
    assert ent.shape == (2, 5)
    expected = math.log(V)
    assert torch.allclose(ent, torch.full((2, 5), expected), atol=1e-3)


def test_entropy_rate_onehot_is_zero():
    """Peaked distribution => entropy near 0."""
    V = 10
    logits = torch.full((1, 3, V), -1e4)
    logits[..., 0] = 1e4  # argmax strongly on class 0
    ent = entropy_rate(logits)
    assert ent.shape == (1, 3)
    assert float(ent.max()) < 1e-3


def test_entropy_rate_no_nan_on_extreme():
    """Extreme logits must not produce NaN/inf."""
    logits = torch.tensor([[[1e9, -1e9, 0.0]]])
    ent = entropy_rate(logits)
    assert torch.isfinite(ent).all()


def test_compression_ratio_arithmetic():
    """train_bytes / model_bytes with fp16 params (2 bytes)."""
    num_params = 53_000_000
    train_tokens = 3_000_000_000
    ratio = compression_ratio(num_params, train_tokens, bytes_per_param=2)
    expected = (3_000_000_000 * 2) / (53_000_000 * 2)
    assert math.isclose(ratio, expected, rel_tol=1e-9)


def test_compression_ratio_default_bytes():
    """Default bytes_per_param=2."""
    ratio = compression_ratio(100, 1000)
    assert math.isclose(ratio, (1000 * 2) / (100 * 2))


def test_calibration_error_perfect_is_zero():
    """Perfectly calibrated: confidence==accuracy in every bin => ECE ~ 0."""
    confidence = torch.ones(100)
    correctness = torch.ones(100)
    ece = calibration_error(confidence, correctness, n_bins=10)
    assert ece < 1e-6


def test_calibration_error_known_value():
    """Constant confidence 0.5 on 80% accuracy => ECE = |0.5 - 0.8| = 0.3."""
    confidence = torch.full((100,), 0.5)
    correctness = torch.cat([torch.ones(80), torch.zeros(20)])
    ece = calibration_error(confidence, correctness, n_bins=10)
    assert math.isclose(ece, 0.3, abs_tol=1e-6)


def test_calibration_error_no_valid_bins():
    """Empty correctness tensor => ECE 0, no crash."""
    ece = calibration_error(torch.empty(0), torch.empty(0), n_bins=10)
    assert ece == 0.0


def test_effective_rank_identity_is_full():
    """Orthogonal matrix => effective rank near full dim."""
    torch.manual_seed(0)
    hidden = torch.randn(64, 64)
    q, _ = torch.linalg.qr(hidden)
    rank = effective_rank(q)
    assert 4.0 < rank <= 64.0


def test_effective_rank_constant_is_one():
    """All-identical rows => rank-1 => effective rank ~ 1.0 (single direction)."""
    hidden = torch.ones(64, 32)
    rank = effective_rank(hidden)
    # Roy-Vetterli: rank-1 => all energy in one singular value => exp(H)=1.0
    assert math.isclose(rank, 1.0, abs_tol=1e-3)


def test_effective_rank_stable_on_small():
    """No NaN/inf on a tiny input."""
    rank = effective_rank(torch.ones(2, 4))
    assert torch.isfinite(torch.tensor(rank)).item()


def test_artifact_ratio_threshold_boundary():
    """Fraction of positions with confidence < threshold."""
    confidence = torch.tensor([0.1, 0.6, 0.4, 0.9, 0.49])
    ratio = artifact_ratio(confidence, threshold=0.5)
    assert math.isclose(ratio, 3 / 5, abs_tol=1e-6)


def test_artifact_ratio_all_confident():
    ratio = artifact_ratio(torch.tensor([0.9, 0.95, 1.0]), threshold=0.5)
    assert ratio == 0.0


def test_artifact_ratio_empty():
    assert artifact_ratio(torch.empty(0), threshold=0.5) == 0.0


def test_monitor_compute_returns_dict():
    """compute() returns float metrics dict from logits/targets."""
    monitor = CompressionMonitor(num_params=1000, train_tokens=100_000, cfg={})
    V = 50
    logits = torch.randn(2, 8, V)
    targets = torch.randint(0, V, (2, 8))
    hidden = torch.randn(16, 32)
    out = monitor.compute(logits=logits, hidden=hidden, targets=targets, step=0)
    assert isinstance(out, dict)
    assert "entropy" in out and isinstance(out["entropy"], float)
    assert "calibration_error" in out
    assert "artifact_ratio" in out
    assert out["compression_ratio"] > 0.0


def test_monitor_heavy_interval_throttles_effective_rank():
    """effective_rank only computed every heavy_interval_mult calls."""
    monitor = CompressionMonitor(
        num_params=1000, train_tokens=100_000, cfg={"heavy_interval_mult": 4}
    )
    logits = torch.randn(2, 8, 50)
    targets = torch.randint(0, 50, (2, 8))
    hidden = torch.randn(16, 32)
    out0 = monitor.compute(logits=logits, hidden=hidden, targets=targets, step=0)
    assert "effective_rank" in out0
    out1 = monitor.compute(logits=logits, hidden=hidden, targets=targets, step=1)
    assert "effective_rank" not in out1


def test_monitor_skips_when_logits_none():
    """None logits (fused_ce) => only compression_ratio, no dependent metrics."""
    monitor = CompressionMonitor(num_params=1000, train_tokens=100_000, cfg={})
    out = monitor.compute(logits=None, hidden=None, targets=None, step=0)
    assert "compression_ratio" in out
    assert "entropy" not in out


def test_hagi_returns_pre_norm_hidden_in_training():
    """HAGI.forward returns pre_norm_hidden key in training_mode result."""
    from hagi.model import HAGI, HAGIConfig, TransformerConfig

    cfg = HAGIConfig(
        vocab_size=64,
        hidden_size=32,
        perception_layers=1,
        reasoning_layers=1,
        expression_layers=1,
        use_msa=False,
        use_moe=False,
        use_gdr=False,
        hrm=False,
        use_nars=False,
        gradient_checkpointing=False,
        transformer=TransformerConfig(
            hidden_size=32,
            num_query_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=64,
        ),
    )
    model = HAGI(cfg)
    model.train()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    out = model(x, targets=y, training_mode=True)
    assert isinstance(out, dict)
    assert "pre_norm_hidden" in out
    assert out["pre_norm_hidden"].shape == (2, 8, 32)
