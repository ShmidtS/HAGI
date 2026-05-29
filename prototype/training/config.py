"""Config loading: YAML -> HAGIConfig + training/data/eval dicts.

Also (de)serializes HAGIConfig to/from a plain nested dict so checkpoints store
the config as primitives (no pickled dataclass) and load under torch's default
`weights_only=True`. See `prototype/training/loop.py` save/load_checkpoint.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from prototype.model.gdr import GradeConfig
from prototype.model.hagi import HAGIConfig
from prototype.model.transformer import TransformerConfig


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)

    m = raw["model"]
    t = m.get("transformer", {})
    tcfg = TransformerConfig(
        hidden_size=m["hidden_size"],
        num_query_heads=t.get("num_query_heads", 12),
        num_kv_heads=t.get("num_kv_heads", 4),
        intermediate_size=t.get("intermediate_size", 2048),
        rope_theta=t.get("rope_theta", 10000.0),
        max_seq_len=t.get("max_seq_len", 4096),
    )

    gcfg = GradeConfig(**m["grades"]) if m.get("use_gdr") and "grades" in m else GradeConfig()

    model_cfg = HAGIConfig(
        vocab_size=m["vocab_size"],
        hidden_size=m["hidden_size"],
        perception_layers=m["perception_layers"],
        reasoning_layers=m["reasoning_layers"],
        expression_layers=m["expression_layers"],
        loop_count=m.get("loop_count", 3),
        use_loop=m.get("use_loop", True),
        use_gdr=m.get("use_gdr", True),
        transformer=tcfg,
        grades=gcfg,
    )

    return {
        "name": raw.get("name", "unnamed"),
        "model": model_cfg,
        "training": raw.get("training", {}),
        "data": raw.get("data", {}),
        "eval": raw.get("eval", {}),
    }


def config_to_dict(cfg: HAGIConfig) -> dict:
    """HAGIConfig -> plain nested dict (primitives only). Safe to torch.save."""
    return dataclasses.asdict(cfg)


def config_from_dict(d: dict) -> HAGIConfig:
    """Inverse of config_to_dict. Reconstructs the nested dataclasses."""
    d = dict(d)
    tcfg = TransformerConfig(**d.pop("transformer"))
    gcfg = GradeConfig(**d.pop("grades"))
    return HAGIConfig(transformer=tcfg, grades=gcfg, **d)
