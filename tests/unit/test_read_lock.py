"""Reading the baseline lock lives in the package; the evaluate command only calls it."""

import json

from click.testing import CliRunner

from taxi_duration import config
from taxi_duration import train as training

LOCK = {
    "metric": "rmse_minutes",
    "fold_scores": [4.1, 4.2, 4.5],
    "mean_rmse": 4.2667,
    "mean_reference_scores": [7.0, 7.1, 7.2],
    "mean_reference_mean_rmse": 7.1,
}


def test_read_lock_returns_contents(tmp_path, monkeypatch):
    lock_file = tmp_path / "lock.json"
    lock_file.write_text(json.dumps(LOCK))

    assert training.read_lock(lock_file) == LOCK
    assert training.read_lock(str(lock_file)) == LOCK

    monkeypatch.setattr(config, "LOCK_PATH", str(lock_file))
    assert training.read_lock() == LOCK


def test_read_lock_errors_name_the_file(tmp_path):
    missing = tmp_path / "missing.json"
    try:
        training.read_lock(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    try:
        training.read_lock(broken)
    except ValueError as exc:
        assert str(broken) in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_evaluate_uses_read_lock(tmp_path, monkeypatch):
    from taxi_duration.cli import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DB_CONN", f"duckdb:///{tmp_path / 'missing.duckdb'}")
    assert not (tmp_path / config.LOCK_PATH).exists()

    sentinel = {**LOCK, "from_read_lock": True}
    monkeypatch.setattr(training, "read_lock", lambda *a, **k: sentinel, raising=False)

    result = CliRunner().invoke(cli, ["evaluate", "--format", "json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["from_read_lock"] is True
