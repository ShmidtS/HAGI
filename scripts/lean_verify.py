#!/usr/bin/env python3
"""Run Lean verification for the HAGI formalization."""

from __future__ import annotations

import sys
from pathlib import Path

from hagi.lean.bridge import verify


def main() -> int:
    formalization = Path(__file__).resolve().parents[1] / "formalization"
    passed = verify(formalization)
    print("Lean verification: PASS" if passed else "Lean verification: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
