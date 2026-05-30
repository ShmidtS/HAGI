import pytest

torch = pytest.importorskip("torch")

from hagi.model import GradeConfig, HAGI, HAGIConfig, TransformerConfig


@pytest.fixture(params=[
    (False, False),
    (True, False),
    (False, True),
    (True, True),
])
def hagi_variant_config(request):
    use_loop, use_gdr = request.param
    return HAGIConfig(
        vocab_size=97,
        hidden_size=64,
        perception_layers=1,
        reasoning_layers=1,
        expression_layers=1,
        loop_count=2,
        use_loop=use_loop,
        use_gdr=use_gdr,
        transformer=TransformerConfig(
            hidden_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            intermediate_size=128,
            max_seq_len=16,
        ),
        grades=GradeConfig(scalar=8, vector=16, bivector=16, trivector=8, residual=16),
    )


def test_hagi_model_instantiates_all_four_variants(hagi_variant_config):
    model = HAGI(hagi_variant_config)

    assert model.cfg.use_loop is hagi_variant_config.use_loop
    assert model.cfg.use_gdr is hagi_variant_config.use_gdr
    assert model.num_parameters() > 0


def test_hagi_forward_pass_logits_shape(hagi_variant_config):
    model = HAGI(hagi_variant_config)
    input_ids = torch.randint(0, hagi_variant_config.vocab_size, (2, 8))

    logits = model(input_ids)

    assert logits.shape == (2, 8, hagi_variant_config.vocab_size)


def test_hagi_loss_computation_when_targets_provided(hagi_variant_config):
    model = HAGI(hagi_variant_config)
    input_ids = torch.randint(0, hagi_variant_config.vocab_size, (2, 8))
    targets = torch.randint(0, hagi_variant_config.vocab_size, (2, 8))

    logits, loss = model(input_ids, targets=targets)

    assert logits.shape == (2, 8, hagi_variant_config.vocab_size)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
