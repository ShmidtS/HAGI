import pytest

torch = pytest.importorskip("torch")

from hagi.model import GradeConfig, HAGI, HAGIConfig, TransformerConfig
from hagi.train.checkpoint import load_checkpoint, save_checkpoint
from hagi.train.loop import LoopConfig, train


@pytest.fixture
def tiny_training_model():
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
            max_seq_len=8,
        ),
        grades=GradeConfig(scalar=8, vector=16, bivector=16, trivector=8, residual=16),
    )
    return HAGI(cfg)


def test_train_for_five_steps_returns_finite_loss_and_checkpoint_roundtrip(tmp_path, tiny_training_model):
    torch.manual_seed(0)
    optimizer = torch.optim.AdamW(tiny_training_model.parameters(), lr=1e-3)
    loop_cfg = LoopConfig(
        max_steps=5,
        warmup_steps=2,
        learning_rate=1e-3,
        precision="fp32",
        log_interval=100,
        eval_interval=0,
        ckpt_interval=0,
        ckpt_dir=str(tmp_path),
    )

    def get_batch():
        x = torch.randint(0, tiny_training_model.cfg.vocab_size, (2, 8))
        y = torch.randint(0, tiny_training_model.cfg.vocab_size, (2, 8))
        return x, y

    loss = train(tiny_training_model, optimizer, get_batch, loop_cfg, device="cpu", on_log=lambda _: None)

    assert torch.isfinite(torch.tensor(loss))

    save_checkpoint(tiny_training_model, optimizer, step=5, ckpt_dir=str(tmp_path))
    loaded_model, step = load_checkpoint(str(tmp_path / "step-00000005.pt"), device="cpu")

    assert step == 5
    for name, tensor in tiny_training_model.state_dict().items():
        assert torch.equal(tensor, loaded_model.state_dict()[name])
