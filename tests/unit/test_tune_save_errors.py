"""Save or disk error in the retrain and in the fallback (ADR 8 point 21), review 3.

Seams: tune_run_support.py. The error is raised only for saves into tune.MODEL_DIR
(the retrain and the fallback; trials save into a temporary folder) and only for the
settings of one attempt.

Interface tune.run_tuning must provide:
- A RuntimeError from artifacts.save_model (check_disk) or a tune._SaveError from
  tune._size_and_speed, inside the retrain of the chosen candidate or inside a fallback
  attempt, counts as a failed size limit for that attempt. run_tuning never lets a bare
  RuntimeError or _SaveError escape from those two places.
- Retrain: the next-best passing candidate is tried (first_passing_after_retrain). When
  none is left, NeedsApproval is raised and trials.json holds "final": null.
- Fallback: the failed attempt is listed in fallback["attempts"] with "disk_error": true
  and "predict_ms": null (same key as a failed trial), and the fallback goes on with the
  next attempt. When the last attempt fails this way the fallback has "misses": true and
  run_tuning raises NeedsApproval whose message says the limits are missed. trials.json
  is written in every one of these cases.
"""

import pytest
from tune_run_support import HUGE, SMALL, params, read_trials, setup_run

BEST = params(depth=12, trees=50)  # mean 3.0, the chosen candidate
SECOND = params(depth=10, trees=60)  # mean 3.2, next best
DEEP_A = params(depth=30, trees=50)  # mean 3.9
DEEP_B = params(depth=20, trees=60)  # mean 3.7, the fallback start
TRIAL_SCORES = {
    (12, 50): 3.0,
    (10, 60): 3.2,
    (30, 50): 3.9,
    (20, 60): 3.7,
}
ERROR_KINDS = ["runtime_error_from_save", "save_error"]


def scores(p):
    return [TRIAL_SCORES.get((p["max_depth"], p["n_estimators"]), 4.0)] * 3


def deep_is_huge(p):
    return HUGE if p["max_depth"] > 12 else SMALL


def never_small(p):
    return HUGE


def same(a, b):
    return {k: a[k] for k in b} == b


def fail_in_model_dir(monkeypatch, kind, should_fail):
    """Raise the chosen error for saves into tune.MODEL_DIR when should_fail(params)."""
    from taxi_duration import artifacts, tune

    raised = []
    if kind == "runtime_error_from_save":
        real_save = artifacts.save_model

        def save(fitted, folder):
            if str(folder) == tune.MODEL_DIR and should_fail(fitted.params):
                raised.append(dict(fitted.params))
                raise RuntimeError("not enough disk space to save the model")
            return real_save(fitted, folder)

        monkeypatch.setattr(artifacts, "save_model", save)
    else:
        real_size = tune._size_and_speed

        def size_and_speed(fitted, row, folder):
            if str(folder) == tune.MODEL_DIR and should_fail(fitted.params):
                raised.append(dict(fitted.params))
                raise tune._SaveError("not enough disk space to save the model")
            return real_size(fitted, row, folder)

        monkeypatch.setattr(tune, "_size_and_speed", size_and_speed)
    return raised


@pytest.mark.parametrize("kind", ERROR_KINDS)
def test_retrain_error_goes_to_the_next_best_passing_candidate(
    monkeypatch, tmp_path, kind
):
    from taxi_duration import tune

    seen = setup_run(monkeypatch, tmp_path, trials=[BEST, SECOND], scores_for=scores)
    raised = fail_in_model_dir(monkeypatch, kind, lambda p: same(p, BEST))

    result = tune.run_tuning()

    assert raised == [BEST]
    assert result["name"] == "trial_1"
    data = read_trials(seen)
    assert data["chosen"] == "trial_0"
    assert data["final"]["name"] == "trial_1"
    assert data["final"]["params"] == SECOND
    assert data["final"]["size_bytes"] == SMALL
    assert "fallback" not in data


@pytest.mark.parametrize("kind", ERROR_KINDS)
def test_retrain_error_for_every_candidate_ends_in_needs_approval(
    monkeypatch, tmp_path, kind
):
    from taxi_duration import tune

    seen = setup_run(monkeypatch, tmp_path, trials=[BEST, SECOND], scores_for=scores)
    raised = fail_in_model_dir(monkeypatch, kind, lambda p: True)

    with pytest.raises(tune.NeedsApproval):
        tune.run_tuning()

    assert raised == [BEST, SECOND]
    data = read_trials(seen)
    assert data["chosen"] == "trial_0"
    assert data["final"] is None
    assert [t["name"] for t in data["trials"]] == ["trial_0", "trial_1"]


@pytest.mark.parametrize("kind", ERROR_KINDS)
def test_fallback_error_moves_on_to_the_next_attempt(monkeypatch, tmp_path, kind):
    from taxi_duration import tune

    seen = setup_run(
        monkeypatch,
        tmp_path,
        trials=[DEEP_A, DEEP_B],
        size_for=deep_is_huge,
        scores_for=scores,
    )
    first_attempt, second_attempt = tune.fallback_attempts(DEEP_B)[:2]
    assert (first_attempt["max_depth"], second_attempt["max_depth"]) == (15, 12)
    raised = fail_in_model_dir(
        monkeypatch, kind, lambda p: p["max_depth"] == 15 and p["n_estimators"] == 60
    )

    with pytest.raises(tune.NeedsApproval) as caught:
        tune.run_tuning()

    assert raised == [first_attempt]
    assert "miss" not in str(caught.value).lower()
    fallback = read_trials(seen)["fallback"]
    tried = fallback["attempts"]
    assert [a["params"] for a in tried] == [first_attempt, second_attempt]
    assert tried[0]["disk_error"] is True
    assert tried[0]["predict_ms"] is None
    assert tried[0]["mean_cv_rmse"] is None
    assert not tried[1].get("disk_error")
    assert tried[1]["size_bytes"] == SMALL
    assert fallback["misses"] is False
    assert fallback["selected"]["params"] == second_attempt
    assert fallback["capped_cv_rmse"] == pytest.approx(4.0)


@pytest.mark.parametrize("kind", ERROR_KINDS)
def test_fallback_whose_last_attempt_fails_this_way_misses(monkeypatch, tmp_path, kind):
    from taxi_duration import tune

    seen = setup_run(
        monkeypatch,
        tmp_path,
        trials=[DEEP_A, DEEP_B],
        size_for=never_small,
        scores_for=scores,
    )
    attempts = tune.fallback_attempts(DEEP_B)
    last = attempts[-1]
    raised = fail_in_model_dir(
        monkeypatch,
        kind,
        lambda p: (
            p["max_depth"] == last["max_depth"]
            and p["n_estimators"] == last["n_estimators"]
        ),
    )

    with pytest.raises(tune.NeedsApproval, match="(?i)miss"):
        tune.run_tuning()

    assert raised == [last]
    fallback = read_trials(seen)["fallback"]
    assert fallback["misses"] is True
    assert [a["params"] for a in fallback["attempts"]] == attempts
    assert fallback["attempts"][-1]["disk_error"] is True
    assert fallback["attempts"][-1]["predict_ms"] is None


@pytest.mark.parametrize("kind", ERROR_KINDS)
def test_fallback_where_every_attempt_fails_this_way_misses(
    monkeypatch, tmp_path, kind
):
    from taxi_duration import tune

    seen = setup_run(
        monkeypatch,
        tmp_path,
        trials=[DEEP_A, DEEP_B],
        size_for=deep_is_huge,
        scores_for=scores,
    )
    attempts = tune.fallback_attempts(DEEP_B)
    raised = fail_in_model_dir(monkeypatch, kind, lambda p: True)

    with pytest.raises(tune.NeedsApproval, match="(?i)miss"):
        tune.run_tuning()

    assert raised == attempts
    data = read_trials(seen)
    assert data["fallback"]["misses"] is True
    assert [a["params"] for a in data["fallback"]["attempts"]] == attempts
    assert all(a["disk_error"] is True for a in data["fallback"]["attempts"])
    assert not data.get("final")
