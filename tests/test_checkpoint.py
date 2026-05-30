import pytest

torch = pytest.importorskip("torch")

from hagi.model import GradeConfig, HAGI, HAGIConfig, TransformerConfig
from hagi.train.checkpoint import load_checkpoint, save_checkpoint


@pytest.fixture
def tiny_model():
    cfg = HAGIConfig(
        vocab_size=32,
        hidden_size=64,
        perception_layers=1,
        reasoning_layers=1,
        expression_layers=1,
        loop_count=2,
        use_loop=True,
        use_gdr=True,
        transformer=TransformerConfig(
            hidden_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            intermediate_size=128,
            max_seq_len=16,
        ),
        grades=GradeConfig(scalar=8, vector=16, bivector=16, trivector=8, residual=16),
    )
    return HAGI(cfg)


def test_save_checkpoint_and_load_back_preserves_state_dict_and_step(tmp_path, tiny_model):
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    save_checkpoint(tiny_model, optimizer, step=7, ckpt_dir=str(tmp_path))

    loaded_model, step = load_checkpoint(str(tmp_path / "step-00000007.pt"), device="cpu")

    assert step == 7
    for name, tensor in tiny_model.state_dict().items():
        assert name in loaded_model.state_dict()
        assert torch.equal(tensor, loaded_model.state_dict()[name])
