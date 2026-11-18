"""run_evaluation order, strict MLflow settings and _chosen_params (review 2, findings
1 and 2). Seams: see evaluate_run_support.py. No fits, no server.

Interface evaluate.py must provide:
- run_evaluation(results_dir) runs these steps in order (it may split them into
  helper functions):
  1. strict settings: the environment URI must be http://127.0.0.1:5001
     (verify_model.check_strict_tracking_uri, raises ValueError) and MODEL_URI must be
     exactly "models:/taxi-duration@champion" (ValueError);
  2. guard: metrics.json present -> FileExistsError; test_read_started.json present
     without metrics.json -> RuntimeError (message names the marker and ADR 8);
  3. read the alias version ONCE (MlflowClient.get_model_version_by_alias), then
     download and verify the champion (verify_model.verify_then_load, once);
  4. only then cv_reports and refit_baseline, then evaluate_test with the loaded
     champion.
  A failure in steps 1 to 3 performs zero fits and zero test reads.
- _chosen_params(results_dir) -> dict. FileNotFoundError when trials.json is missing,
  ValueError when it is not valid JSON or has no final.params dict; both messages name
  trials.json.
"""

import json

import pytest
from evaluate_run_support import MODEL_URI, setup_evaluation

BAD_TRACKING_URIS = [
    "http://localhost:5001",
    "http://[::1]:5001",
    "sqlite:///mlflow.db",
    "file:///tmp/mlruns",
    "https://127.0.0.1:5001",
    "http://127.0.0.1:5000",
    "http://127.0.0.1",
    "http://127.0.0.1:5001/api",
    "http://user@127.0.0.1:5001",
]

BAD_MODEL_URIS = [
    "models:/taxi-duration@baseline",
    "models:/taxi-duration@champion2",
    "models:/taxi-duration/3",
    "models:/taxi-duration/latest",
    "models:/other-model@champion",
]


def run(tmp_path_results):
    from taxi_duration import evaluate

    return evaluate.run_evaluation(tmp_path_results)


def assert_nothing_ran(seen):
    for name in ("cv_reports", "refit_baseline", "load_test"):
        assert seen.count(name) == 0, f"{name} ran: {seen.events}"
    assert not (seen.results / "test_read_started.json").exists()
    assert not (seen.results / "metrics.json").exists()


def test_champion_is_verified_before_any_refit(monkeypatch, tmp_path):
    seen = setup_evaluation(monkeypatch, tmp_path)

    result = run(seen.results)

    assert result["test_reads"] == 1
    assert seen.count("verify") == 1  # loaded once, reused by evaluate_test
    verify = seen.events.index("verify")
    assert verify < seen.events.index("cv_reports")
    assert verify < seen.events.index("refit_baseline")
    assert seen.events.index("refit_baseline") < seen.events.index("load_test")


def test_model_version_comes_from_one_alias_lookup_before_the_download(
    monkeypatch, tmp_path
):
    seen = setup_evaluation(monkeypatch, tmp_path)

    run(seen.results)

    assert seen.count("alias_lookup") == 1
    assert seen.events.index("alias_lookup") < seen.events.index("verify")
    info = json.loads((seen.results / "metrics.json").read_text())["model_info"]
    assert info["version"] == "3"  # the first lookup, the one that was downloaded
    assert info["alias"] == "champion"


def test_rerun_with_metrics_present_performs_zero_fits(monkeypatch, tmp_path):
    seen = setup_evaluation(monkeypatch, tmp_path)
    (seen.results / "metrics.json").write_text(json.dumps({"test_reads": 1}))

    with pytest.raises(FileExistsError):
        run(seen.results)

    for name in ("cv_reports", "refit_baseline", "load_test", "verify"):
        assert seen.count(name) == 0, f"{name} ran: {seen.events}"


def test_marker_without_metrics_refuses_before_any_fit(monkeypatch, tmp_path):
    seen = setup_evaluation(monkeypatch, tmp_path)
    (seen.results / "test_read_started.json").write_text("{}")

    with pytest.raises(RuntimeError, match="test_read_started"):
        run(seen.results)

    for name in ("cv_reports", "refit_baseline", "load_test", "verify"):
        assert seen.count(name) == 0, f"{name} ran: {seen.events}"
    assert not (seen.results / "metrics.json").exists()


def test_model_hash_mismatch_performs_zero_fits(monkeypatch, tmp_path):
    from taxi_duration import verify_model

    error = verify_model.ModelVerificationError("Hash mismatch: pinned x, found y")
    seen = setup_evaluation(monkeypatch, tmp_path, verify_error=error)

    with pytest.raises(verify_model.ModelVerificationError, match="mismatch"):
        run(seen.results)

    assert_nothing_ran(seen)


@pytest.mark.parametrize("uri", BAD_TRACKING_URIS)
def test_bad_environment_uri_loads_and_fits_nothing(monkeypatch, tmp_path, uri):
    seen = setup_evaluation(monkeypatch, tmp_path, tracking_uri=uri)

    with pytest.raises(ValueError):
        run(seen.results)

    assert seen.count("verify") == 0
    assert seen.count("alias_lookup") == 0
    assert_nothing_ran(seen)


@pytest.mark.parametrize("uri", BAD_MODEL_URIS)
def test_model_uri_must_be_exactly_the_champion_alias(monkeypatch, tmp_path, uri):
    seen = setup_evaluation(monkeypatch, tmp_path, model_uri=uri)

    with pytest.raises(ValueError):
        run(seen.results)

    assert seen.count("verify") == 0
    assert_nothing_ran(seen)


def test_the_champion_alias_uri_is_accepted(monkeypatch, tmp_path):
    seen = setup_evaluation(monkeypatch, tmp_path, model_uri=MODEL_URI)

    run(seen.results)

    assert seen.verify_kwargs[0][0] == MODEL_URI


def test_missing_trials_file_stops_before_any_fit(monkeypatch, tmp_path):
    seen = setup_evaluation(monkeypatch, tmp_path, trials=None)

    with pytest.raises(FileNotFoundError, match="trials.json"):
        run(seen.results)

    assert_nothing_ran(seen)


def test_chosen_params_reads_final_params(tmp_path):
    from taxi_duration import evaluate

    (tmp_path / "trials.json").write_text(
        json.dumps({"final": {"params": {"n_estimators": 7}}})
    )

    assert evaluate._chosen_params(tmp_path) == {"n_estimators": 7}


def test_chosen_params_missing_file_is_loud(tmp_path):
    from taxi_duration import evaluate

    with pytest.raises(FileNotFoundError, match="trials.json"):
        evaluate._chosen_params(tmp_path)


@pytest.mark.parametrize(
    "text",
    [
        "not json {",
        "[]",
        "{}",
        json.dumps({"final": None}),
        json.dumps({"final": {}}),
        json.dumps({"final": {"params": None}}),
        json.dumps({"final": {"params": [1, 2]}}),
        json.dumps({"final": "trial_1"}),
    ],
)
def test_chosen_params_unusable_file_is_loud(tmp_path, text):
    from taxi_duration import evaluate

    (tmp_path / "trials.json").write_text(text)

    with pytest.raises(ValueError, match="trials.json"):
        evaluate._chosen_params(tmp_path)
