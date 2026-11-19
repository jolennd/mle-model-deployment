"""make targets for the tuning and evaluation steps (dry run only)."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def dry_run(target):
    return subprocess.run(
        ["make", "-n", target],
        check=False,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_make_targets_tune_and_evaluate():
    tune = dry_run("tune")
    assert tune.returncode == 0, tune.stderr
    assert "uv run" in tune.stdout
    assert " tune" in tune.stdout

    evaluate = dry_run("evaluate")
    assert evaluate.returncode == 0, evaluate.stderr
    assert "uv run" in evaluate.stdout
    assert "evaluate --test-week" in evaluate.stdout
