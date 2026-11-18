"""register_from_results reads trials.json and registers what run_tuning chose.

Interface tune.py must provide:
- register_from_results(tracking_uri=None) -> dict: reads
  <config.RESULTS_DIR>/trials.json (path taken at call time). With a "final" record
  it calls tune.register_champion(final["model_dir"], final["size_bytes"],
  final["predict_ms"], tracking_uri=tracking_uri) once (module global) and returns
  its result.
- FileNotFoundError when trials.json is missing.
- ValueError (message contains "approval") when the file has a "fallback" section:
  the fallback stopped for the human, nothing is registered.
- ValueError (message contains "no model") when there is no usable "final" record.
- register_champion(model_dir, size_bytes, predict_ms, tracking_uri=None) -> dict
  {"name", "alias", "version", "sha256"}; it still prints MODEL_SHA256=<sha>.
"""

import json

import pytest

FINAL = {
    "name": "trial_4",
    "params": {"n_estimators": 80},
    "size_bytes": 123_456,
    "predict_ms": 7.5,
    "model_dir": "models/champion_candidate",
}
REGISTERED = {
    "name": "taxi-duration",
    "alias": "champion",
    "version": "2",
    "sha256": "cd" * 32,
}


@pytest.fixture
def results(monkeypatch, tmp_path):
    from taxi_duration import config

    folder = tmp_path / "results"
    folder.mkdir()
    monkeypatch.setattr(config, "RESULTS_DIR", str(folder))
    return folder


@pytest.fixture
def registered(monkeypatch):
    from taxi_duration import tune

    calls = []

    def register_champion(model_dir, size_bytes, predict_ms, tracking_uri=None):
        calls.append((model_dir, size_bytes, predict_ms, tracking_uri))
        return REGISTERED

    monkeypatch.setattr(tune, "register_champion", register_champion)
    return calls


def write(results, data):
    (results / "trials.json").write_text(json.dumps(data))


def test_registers_the_final_model_from_the_file(results, registered):
    from taxi_duration import tune

    write(results, {"chosen": "trial_4", "final": FINAL, "trials": []})

    got = tune.register_from_results()

    assert got == REGISTERED
    assert registered == [("models/champion_candidate", 123_456, 7.5, None)]

    tune.register_from_results(tracking_uri="http://127.0.0.1:5001")
    assert registered[-1][3] == "http://127.0.0.1:5001"


def test_refuses_when_the_fallback_stopped_for_approval(results, registered):
    from taxi_duration import tune

    write(results, {"chosen": None, "fallback": {"misses": False, "attempts": []}})

    with pytest.raises(ValueError, match="approval"):
        tune.register_from_results()
    assert registered == []


@pytest.mark.parametrize(
    "data",
    [
        {"chosen": None, "trials": []},
        {"chosen": "trial_1", "final": None, "trials": []},
    ],
)
def test_refuses_when_no_model_passed(results, registered, data):
    from taxi_duration import tune

    write(results, data)

    with pytest.raises(ValueError, match="(?i)no model"):
        tune.register_from_results()
    assert registered == []


def test_missing_results_file(results, registered):
    from taxi_duration import tune

    with pytest.raises(FileNotFoundError):
        tune.register_from_results()
    assert registered == []
