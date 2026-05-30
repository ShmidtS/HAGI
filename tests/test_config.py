from pathlib import Path

import pytest

from hagi.model import GradeConfig, HAGIConfig, TransformerConfig
from hagi.train.config import config_from_dict, config_to_dict

yaml = pytest.importorskip("yaml")


@pytest.mark.parametrize("config_name", ["baseline.yaml", "gdr.yaml"])
def test_yaml_configs_have_required_keys(config_name):
    path = Path(__file__).resolve().parents[1] / "configs" / config_name

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert "model" in config
    assert "training" in config
    assert "hidden_size" in config["model"]
    assert "learning_rate" in config["training"]
    assert config["model"]["hidden_size"] > 0
    assert config["training"]["learning_rate"] > 0


def test_config_to_dict_roundtrip_for_hagi_config():
    cfg = HAGIConfig(
        vocab_size=128,
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
            max_seq_len=32,
        ),
        grades=GradeConfig(scalar=8, vector=16, bivector=16, trivector=8, residual=16),
    )

    data = config_to_dict(cfg)

    assert data["hidden_size"] == 64
    assert data["transformer"]["hidden_size"] == 64
    assert data["grades"]["vector"] == 16
    assert config_to_dict(config_from_dict(data)) == data


def test_config_from_dict_rebuilds_hagi_config():
    data = {
        "vocab_size": 256,
        "hidden_size": 64,
        "perception_layers": 1,
        "reasoning_layers": 1,
        "expression_layers": 1,
        "loop_count": 2,
        "use_loop": False,
        "use_gdr": True,
        "transformer": {
            "hidden_size": 64,
            "num_query_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 128,
            "rope_theta": 10000.0,
            "norm_eps": 1e-6,
            "max_seq_len": 32,
        },
        "grades": {
            "scalar": 8,
            "vector": 16,
            "bivector": 16,
            "trivector": 8,
            "residual": 16,
            "scalar_momentum": 0.9,
            "vector_momentum": 0.5,
        },
    }

    cfg = config_from_dict(data)

    assert isinstance(cfg, HAGIConfig)
    assert isinstance(cfg.transformer, TransformerConfig)
    assert isinstance(cfg.grades, GradeConfig)
    assert cfg.hidden_size == 64
    assert cfg.grades.hidden_size == 64
