import pytest

torch = pytest.importorskip("torch")

from hagi.model import GradeConfig, HAGI, HAGIConfig, TransformerConfig
from hagi.model.inference_opt import (
    fold_rmsnorm_into_weights,
    pin_model_weights,
    precompute_rope_tables,
    repack_qkv_for_contiguous,
)


@pytest.fixture
def tiny_model():
    cfg = HAGIConfig(
        vocab_size=32,
        hidden_size=64,
        perception_layers=1,
        reasoning_layers=1,
        expression_layers=1,
        loop_count=1,
        use_loop=False,
        use_gdr=False,
        transformer=TransformerConfig(
            hidden_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            intermediate_size=128,
            max_seq_len=16,
            rope_theta=10000.0,
        ),
    )
    return HAGI(cfg)


def test_fold_rmsnorm_preserves_logits(tiny_model):
    tiny_model.eval()
    x = torch.randint(0, tiny_model.cfg.vocab_size, (1, 8))
    with torch.no_grad():
        baseline = tiny_model(x)

    fold_rmsnorm_into_weights(tiny_model)

    with torch.no_grad():
        folded = tiny_model(x)

    diff = (baseline - folded).abs().max().item()
    assert diff <= 1e-4, f"RMSNorm folding changed logits by {diff:.2e}"


def test_repack_qkv_preserves_logits(tiny_model):
    tiny_model.eval()
    x = torch.randint(0, tiny_model.cfg.vocab_size, (1, 8))
    with torch.no_grad():
        baseline = tiny_model(x)

    repack_qkv_for_contiguous(tiny_model)

    with torch.no_grad():
        repacked = tiny_model(x)

    diff = (baseline - repacked).abs().max().item()
    assert diff <= 1e-5, f"QKV repack changed logits by {diff:.2e}"


def test_precompute_rope_preserves_logits(tiny_model):
    tiny_model.eval()
    x = torch.randint(0, tiny_model.cfg.vocab_size, (1, 8))
    with torch.no_grad():
        baseline = tiny_model(x)

    precompute_rope_tables(tiny_model, max_seq_len=16)

    with torch.no_grad():
        precomputed = tiny_model(x)

    diff = (baseline - precomputed).abs().max().item()
    assert diff <= 1e-5, f"RoPE precompute changed logits by {diff:.2e}"


def test_full_inference_opt_pipeline(tiny_model):
    tiny_model.eval()
    x = torch.randint(0, tiny_model.cfg.vocab_size, (1, 8))
    with torch.no_grad():
        baseline = tiny_model(x)

    fold_rmsnorm_into_weights(tiny_model)
    repack_qkv_for_contiguous(tiny_model)
    precompute_rope_tables(tiny_model, max_seq_len=16)

    with torch.no_grad():
        optimized = tiny_model(x)

    diff = (baseline - optimized).abs().max().item()
    assert diff <= 1e-4, f"Full opt pipeline changed logits by {diff:.2e}"


def test_pin_model_weights_cpu():
    model = torch.nn.Linear(4, 4)
    pin_model_weights(model)
    assert next(model.parameters()).is_contiguous()
