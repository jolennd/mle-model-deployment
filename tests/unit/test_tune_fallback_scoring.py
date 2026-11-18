"""Fallback scoring and the "selected" entry (review 2, findings 7 and 8).

Interface tune.run_tuning must provide in the fallback:
- Every attempt is saved, reloaded and timed (size and speed). Only the SELECTED
  attempt (the first that passes both limits, or the last one when none passes) is
  scored with tune.score_params(sample, attempt): exactly one call per fallback.
- trials.json fallback.attempts still lists each tried attempt with "params",
  "size_bytes", "predict_ms" and "mean_cv_rmse" (None for attempts that were not
  scored, the mean over the 3 folds for the selected one).
- fallback.capped_cv_rmse is the selected attempt's mean CV RMSE.
- fallback["selected"] = {"params", "size_bytes", "predict_ms", "model_dir"} copied
  from the selected attempt; model_dir is tune.MODEL_DIR (it holds that model).
"""

import pytest
from tune_run_support import HUGE, SMALL, params, read_trials, setup_run

DEEP_A = params(depth=30, trees=50)  # mean 3.9
DEEP_B = params(depth=20, trees=60)  # mean 3.7, the best candidate
TRIAL_SCORES = {(30, 50): 3.9, (20, 60): 3.7}


def scores(p):
    return [TRIAL_SCORES.get((p["max_depth"], p["n_estimators"]), 4.0)] * 3


def deep_is_huge(p):
    return HUGE if p["max_depth"] > 12 else SMALL


def never_small(p):
    return HUGE


def fallback_scored(seen, attempts):
    return [p for p in seen.scored if p in attempts]


def test_only_the_selected_attempt_is_scored(monkeypatch, tmp_path):
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

    attempts = tune.fallback_attempts(DEEP_B)[:2]  # depth 15 fails, depth 12 passes
    assert fallback_scored(seen, attempts) == [attempts[1]]
    for attempt in attempts:
        assert attempt in seen.fits  # both were still fitted and measured
    tried = read_trials(seen)["fallback"]["attempts"]
    assert [a["params"] for a in tried] == attempts
    assert [a["size_bytes"] for a in tried] == [HUGE, SMALL]
    assert all(a["predict_ms"] == pytest.approx(1.0) for a in tried)
    assert tried[0]["mean_cv_rmse"] is None
    assert tried[1]["mean_cv_rmse"] == pytest.approx(4.0)


def test_when_nothing_passes_the_last_attempt_is_scored(monkeypatch, tmp_path):
    from taxi_duration import tune

    seen = setup_run(
        monkeypatch,
        tmp_path,
        trials=[DEEP_A, DEEP_B],
        size_for=never_small,
        scores_for=scores,
    )

    with pytest.raises(tune.NeedsApproval):
        tune.run_tuning()

    every = tune.fallback_attempts(DEEP_B)
    assert fallback_scored(seen, every) == [every[-1]]
    fallback = read_trials(seen)["fallback"]
    tried = fallback["attempts"]
    assert [a["params"] for a in tried] == every
    assert [a["mean_cv_rmse"] for a in tried[:-1]] == [None] * (len(every) - 1)
    assert tried[-1]["mean_cv_rmse"] == pytest.approx(4.0)
    assert fallback["capped_cv_rmse"] == pytest.approx(4.0)
    assert fallback["misses"] is True


def test_baseline_start_scores_only_its_first_passing_attempt(monkeypatch, tmp_path):
    from taxi_duration import tune

    seen = setup_run(monkeypatch, tmp_path, trials=[])

    with pytest.raises(tune.NeedsApproval):
        tune.run_tuning()

    first = tune.fallback_attempts({})[0]
    assert seen.scored == [first]  # no trials, so this is the only scoring call
    tried = read_trials(seen)["fallback"]["attempts"]
    assert len(tried) == 1
    assert tried[0]["mean_cv_rmse"] is not None


def test_fallback_records_the_selected_attempt(monkeypatch, tmp_path):
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

    fallback = read_trials(seen)["fallback"]
    selected = fallback["selected"]
    chosen = tune.fallback_attempts(DEEP_B)[1]
    assert selected["params"] == chosen
    assert selected["size_bytes"] == SMALL
    assert selected["predict_ms"] == pytest.approx(1.0)
    assert selected["model_dir"] == tune.MODEL_DIR
    assert fallback["misses"] is False


def test_selected_is_the_last_attempt_when_the_limits_are_missed(monkeypatch, tmp_path):
    from taxi_duration import tune

    seen = setup_run(
        monkeypatch,
        tmp_path,
        trials=[DEEP_A, DEEP_B],
        size_for=never_small,
        scores_for=scores,
    )

    with pytest.raises(tune.NeedsApproval):
        tune.run_tuning()

    fallback = read_trials(seen)["fallback"]
    selected = fallback["selected"]
    assert selected["params"] == tune.fallback_attempts(DEEP_B)[-1]
    assert selected["size_bytes"] == HUGE
    assert selected["model_dir"] == tune.MODEL_DIR
    assert fallback["misses"] is True
