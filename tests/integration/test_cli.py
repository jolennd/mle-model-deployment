"""Command-line interface: train and evaluate, run in-process in a temporary folder."""

import json

import duckdb
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from taxi_duration import config

pytestmark = [pytest.mark.integration, pytest.mark.cli]


def make_trips(n=600):
    rng = np.random.default_rng(7)
    distance = rng.uniform(0.5, 5, n)
    minutes = 5 + 3 * distance + rng.normal(0, 0.5, n)
    pickup = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(n) * 55, unit="m")
    return pd.DataFrame(
        {
            "trip_id": [f"t{i:04d}" for i in range(n)],
            "pickup_at": pickup,
            "dropoff_at": pickup + pd.to_timedelta(minutes, unit="m"),
            "pickup_zone_id": rng.choice([10, 20, 30], n),
            "dropoff_zone_id": rng.choice([40, 50], n),
            "trip_distance": distance,
        }
    )


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def work(tmp_path, monkeypatch, db_conn):
    """Temporary folder with a small mart_trips table and a local model store."""
    monkeypatch.chdir(tmp_path)
    trips = make_trips()
    con = duckdb.connect(str(db_conn))
    con.register("trips_df", trips)
    con.execute("CREATE TABLE mart_trips AS SELECT * FROM trips_df")
    con.close()
    # A sqlite store in the environment is refused (ADR 8 point 19): log nothing.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return tmp_path


def cli_group():
    from taxi_duration.cli import cli

    return cli


TRAIN = ["train", "--sample-size", "400", "--n-estimators", "5", "--format", "json"]


def test_train_and_evaluate_commands(runner, work, monkeypatch):
    cli = cli_group()
    result = runner.invoke(cli, TRAIN)
    assert result.exit_code == 0, result.output

    printed = json.loads(result.stdout)
    for key in ("mean_rmse", "mean_reference_mean_rmse", "fold_scores", "lock_path"):
        assert key in printed
    assert len(printed["fold_scores"]) == len(config.CV_FOLDS)

    lock_file = work / config.LOCK_PATH
    assert lock_file.is_file()
    lock_bytes = lock_file.read_bytes()
    lock = json.loads(lock_bytes)
    assert lock["mean_rmse"] == printed["mean_rmse"]

    # evaluate reads the lock only: point the database at a file that does not exist.
    monkeypatch.setenv("DB_CONN", f"duckdb:///{work / 'missing.duckdb'}")
    evaluated = runner.invoke(cli, ["evaluate", "--format", "json"])
    assert evaluated.exit_code == 0, evaluated.output
    assert json.loads(evaluated.stdout) == lock

    text = runner.invoke(cli, ["evaluate"])
    assert text.exit_code == 0, text.output
    assert text.stdout.strip()

    # a second train never overwrites the lock
    again = runner.invoke(cli, TRAIN)
    assert again.exit_code == 1, again.output
    assert isinstance(again.exception, SystemExit)
    assert again.stderr.strip()
    assert "Traceback" not in again.output
    assert lock_file.read_bytes() == lock_bytes


def test_cli_errors_and_help(runner, work):
    cli = cli_group()
    for args in (["--help"], ["train", "--help"], ["evaluate", "--help"]):
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output
    assert runner.invoke(cli, ["--version"]).exit_code == 0

    missing = runner.invoke(cli, ["evaluate"])
    assert missing.exit_code == 1, missing.output
    assert isinstance(missing.exception, SystemExit)
    assert len(missing.stderr.strip().splitlines()) == 1
    assert "Traceback" not in missing.output

    assert runner.invoke(cli, ["no-such-command"]).exit_code == 2
    bad = runner.invoke(cli, ["train", "--sample-size", "many"])
    assert bad.exit_code == 2, bad.output
    assert not (work / config.LOCK_PATH).exists()

    # results on stdout, logs on stderr: stdout is pure json
    trained = runner.invoke(cli, TRAIN)
    assert trained.exit_code == 0, trained.output
    json.loads(trained.stdout)
