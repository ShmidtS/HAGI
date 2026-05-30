from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LeanBridge:
    lake_command: str = "lake"

    def verify(self, root_path: str | Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.lake_command, "build"],
            cwd=Path(root_path),
            text=True,
            capture_output=True,
            check=False,
        )

    def check(self, root_path: str | Path) -> bool:
        return self.verify(root_path).returncode == 0


def verify(root_path: str | Path) -> bool:
    return LeanBridge().check(root_path)
