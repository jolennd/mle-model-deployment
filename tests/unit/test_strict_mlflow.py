"""Strict MLflow address (ADR 8 point 19), review 2 finding 2.

Interface:
- verify_model.check_strict_tracking_uri(uri) -> uri. Accepts only
  http://127.0.0.1:5001, optionally with a trailing "/" (scheme http, hostname
  127.0.0.1, port 5001, no user info, path empty or "/"). Everything else raises
  ValueError. tune._check_uri is the same check (same rules, same exception).
- It applies where the URI is read from the environment: tune.run_tuning,
  tune.register_from_results, evaluate.run_evaluation (see test_evaluate_order.py) and
  verify_model.main. An explicit tracking_uri argument (register_champion,
  register_from_results, run_tuning, verify_then_load, download) is NOT checked by
  this rule, so a sqlite store in tmp_path stays usable in tests.
- On a bad environment URI nothing is fitted, registered, downloaded or loaded.
"""

import json

import pytest
from tune_run_support import SMALL, params, read_trials, setup_run

GOOD = ["http://127.0.0.1:5001", "http://127.0.0.1:5001/"]
BAD = [
    "http://localhost:5001",
    "http://[::1]:5001",
    "sqlite:///mlflow.db",
    "file:///tmp/mlruns",
    "https://127.0.0.1:5001",
    "http://127.0.0.1:5000",
    "http://127.0.0.1:5001:80",
    "http://127.0.0.1",
    "http://127.0.0.1/",
    "http://user@127.0.0.1:5001",
    "http://user" + ":pw@127.0.0.1:5001",
    "http://127.0.0.1@evil.com:5001",
    "http://127.0.0.1.evil.com:5001",
    "http://127.0.0.1:5001/api",
    "127.0.0.1:5001",
    "",
]


@pytest.mark.parametrize("uri", GOOD)
def test_strict_check_accepts_only_the_local_server(uri):
    from taxi_duration import tune, verify_model

    assert verify_model.check_strict_tracking_uri(uri) == uri
    assert tune._check_uri(uri) == uri


@pytest.mark.parametrize("uri", BAD)
def test_strict_check_refuses_everything_else(uri):
    from taxi_duration import tune, verify_model

    with pytest.raises(ValueError):
        verify_model.check_strict_tracking_uri(uri)
    with pytest.raises(ValueError):
        tune._check_uri(uri)


@pytest.mark.parametrize("uri", [u for u in BAD if u])
def test_run_tuning_refuses_a_bad_environment_uri(monkeypatch, tmp_path, uri):
    from taxi_duration import tune

    seen = setup_run(monkeypatch, tmp_path, trials=[params(depth=12)])
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    with pytest.raises(ValueError):
        tune.run_tuning()

    assert seen.experiment_calls == []
    assert seen.client.runs == []
    assert seen.fits == []
    assert seen.baseline_timed == []
    assert seen.saves == []
    assert not (seen.results / "trials.json").exists()


def test_run_tuning_accepts_the_local_server(monkeypatch, tmp_path):
    from taxi_duration import tune

    seen = setup_run(monkeypatch, tmp_path, trials=[params(depth=12)])
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001/")

    assert tune.run_tuning()["size_bytes"] == SMALL
    assert read_trials(seen)["final"]["name"] == "trial_0"


@pytest.fixture
def saved_final(monkeypatch, tmp_path):
    """A results folder with a usable final record and its model file."""
    from taxi_duration import config, tune

    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(config, "RESULTS_DIR", str(results))
    model_dir = tmp_path / "candidate"
    model_dir.mkdir()
    (model_dir / "model.skops").write_bytes(b"x" * 1_000)
    final = {
        "name": "trial_1",
        "params": {},
        "size_bytes": 1_000,
        "predict_ms": 2.0,
        "model_dir": str(model_dir),
    }
    (results / "trials.json").write_text(json.dumps({"final": final}))
    calls = []
    monkeypatch.setattr(
        tune, "register_champion", lambda *a, **k: calls.append((a, k)) or {}
    )
    return calls


@pytest.mark.parametrize("uri", [u for u in BAD if u])
def test_register_from_results_refuses_a_bad_environment_uri(
    monkeypatch, saved_final, uri
):
    from taxi_duration import tune

    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    with pytest.raises(ValueError):
        tune.register_from_results()

    assert saved_final == []


def test_register_from_results_with_the_local_server_registers(
    monkeypatch, saved_final
):
    from taxi_duration import tune

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")

    tune.register_from_results()

    assert len(saved_final) == 1


@pytest.mark.parametrize("uri", [u for u in BAD if u])
def test_verify_step_refuses_a_bad_environment_uri(monkeypatch, uri):
    from taxi_duration import verify_model

    reached = []

    def download(*args, **kwargs):
        reached.append(1)
        raise RuntimeError("download reached")

    monkeypatch.setattr(verify_model, "download", download)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    with pytest.raises(ValueError):
        verify_model.main(["models:/taxi-duration@champion"])

    assert reached == []


def test_verify_step_refuses_when_the_environment_uri_is_unset(monkeypatch):
    from taxi_duration import verify_model

    reached = []

    def download(*args, **kwargs):
        reached.append(1)
        raise RuntimeError("download reached")

    monkeypatch.setattr(verify_model, "download", download)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        verify_model.main(["models:/taxi-duration@champion"])

    assert reached == []
