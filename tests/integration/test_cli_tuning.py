"""CLI: tune, and evaluate --test-week. The package calls are stubbed.

Interface to implement in cli.py (modules imported as modules, so tests can patch):
- tune [--format text|json]: calls tune.run_tuning() once, prints the result
  (name, size_bytes, predict_ms); RuntimeError becomes a ClickException (exit 1).
- evaluate [--test-week] [--format text|json]: with the flag calls
  evaluate.run_evaluation() once and prints chosen and baseline rmse; a
  FileExistsError, ValueError or RuntimeError gives exit 1. Without the flag it
  only reads the lock.
"""

import json

import pytest
from click.testing import CliRunner

pytestmark = [pytest.mark.integration, pytest.mark.cli]

TUNED = {
    "name": "trial_3",
    "params": {"n_estimators": 80},
    "size_bytes": 123456,
    "predict_ms": 7.5,
    "model_dir": "models/champion_candidate",
}
EVALUATED = {
    "chosen": {"rmse": 4.5},
    "baseline": {"rmse": 6.25},
    "rows_used": 10,
    "test_reads": 1,
}


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def cli_group():
    from taxi_duration.cli import cli

    return cli


def clean_failure(result, message):
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert "Error:" in result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.output


def stub_tune(monkeypatch, outcome):
    from taxi_duration import tune

    calls = []

    def run_tuning(*args, **kwargs):
        calls.append(1)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(tune, "run_tuning", run_tuning)
    return calls


def stub_evaluation(monkeypatch, outcome):
    from taxi_duration import evaluate

    calls = []

    def run_evaluation(*args, **kwargs):
        calls.append(1)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(evaluate, "run_evaluation", run_evaluation)
    return calls


def test_cli_tune_and_evaluate_test_week(runner, work, monkeypatch):
    cli = cli_group()

    calls = stub_tune(monkeypatch, TUNED)
    text = runner.invoke(cli, ["tune"])
    assert text.exit_code == 0, text.output
    assert calls == [1]
    assert "trial_3" in text.stdout
    as_json = runner.invoke(cli, ["tune", "--format", "json"])
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.stdout)["name"] == "trial_3"
    assert json.loads(as_json.stdout)["size_bytes"] == 123456

    stub_tune(monkeypatch, RuntimeError("No candidate passes the limits."))
    failed = runner.invoke(cli, ["tune"])
    clean_failure(failed, "No candidate passes the limits.")

    calls = stub_evaluation(monkeypatch, EVALUATED)
    text = runner.invoke(cli, ["evaluate", "--test-week"])
    assert text.exit_code == 0, text.output
    assert calls == [1]
    assert "4.5" in text.stdout
    assert "6.25" in text.stdout
    assert "chosen" in text.stdout.lower()
    assert "baseline" in text.stdout.lower()
    as_json = runner.invoke(cli, ["evaluate", "--test-week", "--format", "json"])
    assert as_json.exit_code == 0, as_json.output
    printed = json.loads(as_json.stdout)
    assert printed["chosen"]["rmse"] == 4.5
    assert printed["baseline"]["rmse"] == 6.25
    assert calls == [1, 1]

    stub_evaluation(monkeypatch, FileExistsError("metrics file already exists: m"))
    again = runner.invoke(cli, ["evaluate", "--test-week"])
    clean_failure(again, "metrics file already exists")


def test_evaluate_without_flag_reads_only_the_lock(runner, work, monkeypatch):
    from taxi_duration import train

    lock = {
        "fold_scores": [4.0, 3.9, 3.8],
        "mean_rmse": 3.9,
        "mean_reference_mean_rmse": 7.0,
    }
    monkeypatch.setattr(train, "read_lock", lambda *a, **k: lock)
    calls = stub_evaluation(monkeypatch, EVALUATED)

    result = runner.invoke(cli_group(), ["evaluate", "--format", "json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == lock
    assert calls == []
