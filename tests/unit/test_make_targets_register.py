"""make register target (dry run only)."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_make_target_register():
    result = subprocess.run(
        ["make", "-n", "register"],
        check=False,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "uv run" in result.stdout
    assert "taxi_duration register" in result.stdout
