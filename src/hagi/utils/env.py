"""Load environment variables from the project root .env file.

Single source of truth for .env parsing. All modules that need HF_TOKEN
or other env vars import from here instead of duplicating the parsing.

Usage (at module top, before HF imports):
    from hagi.utils.env import load_env
    load_env()
"""

from __future__ import annotations

import os
from pathlib import Path


_loaded = False


def _project_root() -> Path:
    """Walk up from this file to find the project root (contains .env)."""
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".env").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_env() -> None:
    """Load all KEY=VALUE pairs from .env into os.environ.

    Idempotent: only runs once per process. Does NOT overwrite existing
    env vars (os.environ.setdefault), so explicit exports win.

    Strips surrounding quotes from values (supports 'val' and "val").
    Skips blank lines and comments (#).
    """
    global _loaded
    if _loaded:
        return
    _loaded = True

    env_path = _project_root() / ".env"
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                os.environ.setdefault(key, value)
