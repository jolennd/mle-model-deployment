"""run_tuning behavior (batch 4): lock check, fallback, rerun guard, incremental
write and a clean model folder before every save.

Interface tune.py must provide:
- run_tuning compares train.fingerprint(sample["trip_id"]) with lock["sample_sha256"]
  and train.fold_fingerprints(folds) with lock["folds"] before any fit; a mismatch
  raises ValueError naming sample_sha256 or folds.
- class NeedsApproval(RuntimeError). Raised after the fallback ran (message says it
  stops for the human; when no capped model passes the message says the limits are
  missed). Nothing is registered.
- trials.json gets a "fallback" section only when the fallback ran:
  {"from": candidate name, "attempts": [{"params", "size_bytes", "predict_ms",
  "mean_cv_rmse"}, ...] (tried attempts, in order), "skipped_caps": [...],
  "capped_cv_rmse": mean CV RMSE of the first passing (or last tried) attempt,
  "uncapped_cv_rmse": mean CV RMSE of the candidate, "misses": bool}.
  Only the selected attempt is scored through tune.score_params(sample, params); the
  other tried attempts have "mean_cv_rmse": None (see test_tune_fallback_scoring.py).
  "skipped_caps" is absent or empty when the fallback did not run.
- trials.json is rewritten after every completed trial. If it exists when
  run_tuning starts, FileExistsError is raised before any MLflow call.
- prepare_model_dir(path) is called (through the module global) before every
  artifacts.save_model call that run_tuning makes, with the same folder.
"""

import json

import pytest
from tune_run_support import (
    HUGE,
    SMALL,
    params,
    read_trials,
    setup_run,
)

GOOD = params(depth=12)
DEEP_A = params(depth=30, trees=50)  # mean 3.9
DEEP_B = params(depth=20, trees=60)  # mean 3.7, the best candidate
TRIAL_SCORES = {(30, 50): 3.9, (20, 60): 3.7}


def scores(p):
    return [TRIAL_SCORES.get((p["max_depth"], p["n_estimators"]), 4.0)] * 3


def deep_is_huge(p):
    return HUGE if p["max_depth"] > 12 else SMALL


def never_small(p):
    return HUGE


def test_needs_approval_is_a_runtime_error():
    from taxi_duration import tune

    assert issubclass(tune.NeedsApproval, RuntimeError)


@pytest.mark.parametrize("which", ["sample_sha256", "folds"])
def test_lock_mismatch_stops_before_any_fit(monkeypatch, tmp_path, which):
    from taxi_duration import tune

    def edit(lock):
        if which == "sample_sha256":
            lock["sample_sha256"] = "0" * 64
        else:
            lock["folds"][1] = {**lock["folds"][1], "valid_sha256": "0" * 64}

    seen = setup_run(monkeypatch, tmp_path, trials=[GOOD], edit_lock=edit)

    with pytest.raises(ValueError, match=which):
        tune.run_tuning()

    assert seen.fits == []
    assert seen.baseline_timed == []
    assert seen.scored == []
    assert seen.saves == []
    assert seen.client.runs == []
    assert not (seen.results / "trials.json").exists()


def test_fallback_runs_scores_capped_models_and_stops(monkeypatch, tmp_path):
    from taxi_duration import tune

    seen = setup_run(
        monkeypatch,
        tmp_path,
        trials=[DEEP_A, DEEP_B],
        size_for=deep_is_huge,
        scores_for=scores,
    )

    with pytest.raises(tune.NeedsApproval) as caught:
        tune.run_tuning()

    assert "human" in str(caught.value).lower()
    assert len(seen.client.runs) == 2  # the two trials only, no model version
    data = read_trials(seen)
    assert not data.get("final")
    fallback = data["fallback"]
    assert fallback["from"] == "trial_1"  # best mean CV RMSE, whatever the limits
    assert fallback["skipped_caps"] == [25, 20]  # ADR 8 point 13
    tried = fallback["attempts"]
    assert [a["params"]["max_depth"] for a in tried] == [15, 12]  # first pass stops
    assert [a["params"]["n_estimators"] for a in tried] == [60, 60]
    assert tried[0]["size_bytes"] == HUGE
    assert tried[1]["size_bytes"] == SMALL
    # only the selected attempt gets the 3-fold CV (finding 7)
    assert tried[0]["mean_cv_rmse"] is None
    assert tried[1]["mean_cv_rmse"] == pytest.approx(4.0)
    assert fallback["capped_cv_rmse"] == pytest.approx(4.0)
    assert fallback["uncapped_cv_rmse"] == pytest.approx(3.7)
    assert fallback["misses"] is False
    # each tried capped model was retrained and scored on the folds
    expected = tune.fallback_attempts(DEEP_B)[:2]
    for attempt in expected:
        assert attempt in seen.fits
    assert expected[1] in seen.scored  # the selected attempt only


def test_fallback_that_finds_no_model_says_the_limits_are_missed(monkeypatch, tmp_path):
    from taxi_duration import tune

    seen = setup_run(
        monkeypatch,
        tmp_path,
        trials=[DEEP_A, DEEP_B],
        size_for=never_small,
        scores_for=scores,
    )

    with pytest.raises(tune.NeedsApproval, match="(?i)miss"):
        tune.run_tuning()

    fallback = read_trials(seen)["fallback"]
    assert fallback["misses"] is True
    every = tune.fallback_attempts(DEEP_B)
    assert [a["params"] for a in fallback["attempts"]] == every
    assert len(every) == 6
    assert fallback["skipped_caps"] == [25, 20]


def test_success_has_no_fallback_section(monkeypatch, tmp_path):
    from taxi_duration import tune

    seen = setup_run(monkeypatch, tmp_path, trials=[GOOD])

    result = tune.run_tuning()

    assert result["name"] == "trial_0"
    data = read_trials(seen)
    assert "fallback" not in data
    assert not data.get("skipped_caps")  # only recorded when the fallback ran
    assert data["final"]["model_dir"] == tune.MODEL_DIR
    assert data["final"]["size_bytes"] == SMALL
    assert data["final"]["predict_ms"] == pytest.approx(1.0)


def test_existing_trials_file_blocks_the_rerun(monkeypatch, tmp_path):
    from taxi_duration import tune

    seen = setup_run(monkeypatch, tmp_path, trials=[GOOD])
    seen.results.mkdir(parents=True)
    old = seen.results / "trials.json"
    old.write_text(json.dumps({"keep": "me"}))

    with pytest.raises(FileExistsError):
        tune.run_tuning()

    assert seen.client.runs == []
    assert seen.experiment_calls == []
    assert seen.fits == []
    assert json.loads(old.read_text()) == {"keep": "me"}


def test_trials_file_is_written_after_each_trial(monkeypatch, tmp_path):
    """Seam: tune.run_study calls the objective per trial; tune.score_params fails
    in trial 2 after trial 1 completed."""
    from taxi_duration import tune

    def crash_in_second(p):
        if p == DEEP_B:
            raise RuntimeError("boom in trial 2")
        return [3.0, 3.0, 3.0]

    seen = setup_run(
        monkeypatch,
        tmp_path,
        trials=[GOOD, DEEP_B],
        scores_for=crash_in_second,
    )

    with pytest.raises(RuntimeError, match="boom in trial 2"):
        tune.run_tuning()

    data = read_trials(seen)
    assert [t["name"] for t in data["trials"]] == ["trial_0"]


def assert_each_save_was_prepared(seen):
    assert seen.saves, "no model was saved"
    assert seen.unprepared_saves == []
    assert len(seen.prepares) >= len(seen.saves)


def test_model_folder_is_cleared_before_every_save_on_success(monkeypatch, tmp_path):
    from taxi_duration import tune

    seen = setup_run(monkeypatch, tmp_path, trials=[GOOD])

    tune.run_tuning()

    assert len(seen.saves) >= 2  # fold 3 of the trial, then the retrain
    assert tune.MODEL_DIR in seen.saves
    assert_each_save_was_prepared(seen)


def test_model_folder_is_cleared_before_every_save_in_the_fallback(
    monkeypatch, tmp_path
):
    from taxi_duration import tune

    seen = setup_run(
        monkeypatch,
        tmp_path,
        trials=[DEEP_A, DEEP_B],
        size_for=deep_is_huge,
        scores_for=scores,
    )

    with pytest.raises(tune.NeedsApproval):
        tune.run_tuning()

    assert len(seen.saves) >= 4  # two trials at fold 3, two capped attempts
    assert_each_save_was_prepared(seen)


def test_baseline_can_be_the_fallback_start(monkeypatch, tmp_path):
    """Zero trials: the baseline (params {}) is the best mean and fails the size limit
    (BASELINE_SIZE_BYTES). Its settings are the scikit-learn defaults."""
    from taxi_duration import tune

    seen = setup_run(monkeypatch, tmp_path, trials=[])

    with pytest.raises(tune.NeedsApproval):
        tune.run_tuning()

    fallback = read_trials(seen)["fallback"]
    assert fallback["from"] == "baseline"
    first = fallback["attempts"][0]["params"]
    assert first["n_estimators"] == 100
    assert first["max_depth"] == 25
    assert seen.fits[0]["n_estimators"] == 100
