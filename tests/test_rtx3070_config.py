from pathlib import Path

import pytest

from hagi.train.config import config_from_dict

yaml = pytest.importorskip("yaml")


def load_rtx3070_config():
    path = Path(__file__).resolve().parents[1] / "configs" / "rtx3070_canonical.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_rtx3070_yaml_matches_8gb_model_shape():
    cfg = load_rtx3070_config()
    model_cfg = cfg["model"]
    transformer = model_cfg["transformer"]
    grades = model_cfg["grades"]

    assert cfg["name"] == "rtx3070_canonical"
    assert model_cfg["vocab_size"] == 49152
    assert model_cfg["hidden_size"] == 768
    assert transformer["hidden_size"] == 768
    assert transformer["num_query_heads"] == 12
    assert transformer["num_kv_heads"] == 4
    assert transformer["intermediate_size"] == 2048
    assert transformer["max_seq_len"] == 2048
    assert sum(v for v in grades.values() if isinstance(v, int)) == 768


def test_rtx3070_config_builds_model_with_gradient_checkpointing():
    pytest.importorskip("torch")
    from hagi.model import HAGI, HAGIConfig

    cfg = load_rtx3070_config()
    model_cfg = config_from_dict(cfg["model"])
    model = HAGI(model_cfg)

    assert isinstance(model_cfg, HAGIConfig)
    assert model.cfg.gradient_checkpointing is True
    assert model.embed.weight.shape == (49152, 768)
    assert model.lm_head.weight.shape == (49152, 768)
    assert len(model.perception) == 4
    assert len(model.reasoning) == 4
    assert len(model.expression) == 4


def test_rtx3070_canonical_mix_paths_match_mix_weights():
    path = Path(__file__).resolve().parents[1] / "configs" / "rtx3070_canonical.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    data = config["data"]
    mix = data["mix"]
    mix_paths = data["mix_paths"]

    assert data["dataset_mode"] == "memmap_packed"
    assert {entry["name"] for entry in mix_paths} == set(mix)
    assert sum(entry["weight"] for entry in mix_paths) == pytest.approx(1.0)
    for entry in mix_paths:
        assert entry["weight"] == pytest.approx(mix[entry["name"]])
        assert entry["path"].endswith(".bin")


def test_gradient_checkpointing_forward_flag_runs_on_tiny_model():
    torch = pytest.importorskip("torch")
    from hagi.model import HAGI

    cfg = config_from_dict({
        "vocab_size": 64,
        "hidden_size": 64,
        "perception_layers": 1,
        "reasoning_layers": 1,
        "expression_layers": 1,
        "loop_count": 2,
        "use_loop": True,
        "use_gdr": True,
        "gradient_checkpointing": True,
        "transformer": {
            "hidden_size": 64,
            "num_query_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 128,
            "max_seq_len": 8,
        },
        "grades": {"scalar": 8, "vector": 16, "bivector": 16, "trivector": 8, "residual": 16},
    })
    model = HAGI(cfg)
    model.train()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))

    logits, loss = model(input_ids, targets=targets)
    loss.backward()

    assert logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(loss)
    assert model.embed.weight.grad is not None
