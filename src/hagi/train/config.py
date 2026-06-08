"""Training config serialization helpers."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain nested dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
    from hagi.model import GradeConfig, HAGIConfig, TransformerConfig

    values = dict(d)
    transformer = values.get("transformer")
    grades = values.get("grades")

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

    return HAGIConfig(**values)
