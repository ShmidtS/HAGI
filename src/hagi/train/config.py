"""Training config serialization helpers."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast


def config_to_dict(cfg: Any) -> Any:
    """Convert a config object into a plain nested dictionary."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return {
            field.name: config_to_dict(getattr(cfg, field.name))
            for field in dataclasses.fields(cfg)
        }
    if isinstance(cfg, Mapping):
        return {str(key): config_to_dict(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [config_to_dict(value) for value in cfg]
    return cfg


def config_from_dict(d: Mapping[str, Any]):
    """Rebuild HAGIConfig from a plain nested dictionary."""
    from hagi.model import CASTConfig, GradeConfig, HAGIConfig, TransformerConfig

    values = dict(d)
    transformer = values.get("transformer")
    grades = values.get("grades")
    cast_cfg = values.get("cast_config")

    if isinstance(transformer, Mapping):
        assert isinstance(transformer, dict)
        values["transformer"] = TransformerConfig(**cast(dict[str, Any], transformer))
    elif transformer is None:
        values.pop("transformer", None)

    if isinstance(grades, Mapping):
        assert isinstance(grades, dict)
        values["grades"] = GradeConfig(**cast(dict[str, Any], grades))
    elif grades is None:
        values.pop("grades", None)

    if isinstance(cast_cfg, Mapping):
        assert isinstance(cast_cfg, dict)
        values["cast_config"] = CASTConfig(**cast(dict[str, Any], cast_cfg))
    elif cast_cfg is None:
        values.pop("cast_config", None)

    return HAGIConfig(**values)
