import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

bridge_module = pytest.importorskip("hagi.lean.bridge")
LeanBridge = getattr(bridge_module, "LeanBridge", None)
if LeanBridge is None:
    pytest.skip("LeanBridge is not implemented", allow_module_level=True)


def test_lean_bridge_verify_with_mocked_subprocess(tmp_path):
    bridge = LeanBridge(lake_command="lake")
    completed = subprocess.CompletedProcess(
        args=["lake", "build"],
        returncode=0,
        stdout="verified",
        stderr="",
    )

    with patch("subprocess.run", return_value=completed) as run:
        result = bridge.verify(tmp_path)

    run.assert_called_once_with(
        ["lake", "build"],
        cwd=Path(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "verified" in result.stdout

    with patch("subprocess.run", return_value=completed):
        assert bridge.check(tmp_path) is True


def test_lean_bridge_verify_timeout_handling(tmp_path):
    bridge = LeanBridge(lake_command="lake")

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["lake", "build"], timeout=1)):
        with pytest.raises(subprocess.TimeoutExpired):
            bridge.verify(tmp_path)


def test_lean_bridge_missing_lake_executable_handling(tmp_path):
    bridge = LeanBridge(lake_command="missing-lake")

    with patch("subprocess.run", side_effect=FileNotFoundError("missing-lake")):
        with pytest.raises(FileNotFoundError, match="missing-lake"):
            bridge.verify(tmp_path)
