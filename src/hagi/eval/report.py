from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def write_json_report(results: Mapping[str, float], path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(dict(results), indent=2, sort_keys=True), encoding="utf-8")


def write_text_report(results: Mapping[str, float], path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name}: {value:.6g}" for name, value in sorted(results.items())]
    report_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
