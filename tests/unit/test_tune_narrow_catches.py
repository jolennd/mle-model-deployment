"""Narrow catches in the retrain and the fallback (review 4, item 2).

Interface tune.run_tuning must provide: only tune._SaveError (what _size_and_speed
raises for a failed save or disk check, wrapping a RuntimeError from
artifacts.save_model) counts as a failed limit. Any other RuntimeError from
artifacts.load_model or tune.measure_predict_ms, inside the retrain or a fallback
attempt, propagates out of run_tuning as that error (not NeedsApproval). It is not
recorded as disk_error anywhere. Whether trials.json is rewritten on the way out is
not tested.
"""

import pytest
from tune_retrain_support import control_model_dir
from tune_run_support import HUGE, SMALL, params, setup_run

BEST = params(depth=12, trees=50)
SECOND = params(depth=10, trees=60)
DEEP_A = params(depth=30, trees=50)
DEEP_B = params(depth=20, trees=60)
SCORES = {(12, 50): 3.0, (10, 60): 3.2, (30, 50): 3.9, (20, 60): 3.7}
KINDS = ["load_error", "ms_error"]


def scores(p):
    return [SCORES.get((p["max_depth"], p["n_estimators"]), 4.0)] * 3


def deep_is_huge(p):
    return HUGE if p["max_depth"] > 12 else SMALL


def assert_plain_error(caught, seen):
    from taxi_duration import tune

    assert not isinstance(caught.value, tune.NeedsApproval)
    assert "model file is corrupt" in str(caught.value)
    path = seen.results / "trials.json"
    if path.exists():
        assert "disk_error" not in path.read_text()


@pytest.mark.parametrize("kind", KINDS)
def test_retrain_non_save_error_propagates(monkeypatch, tmp_path, kind):
    from taxi_duration import tune

    seen = setup_run(monkeypatch, tmp_path, trials=[BEST, SECOND], scores_for=scores)
    control_model_dir(
        monkeypatch, {(12, 50): {kind: RuntimeError("model file is corrupt")}}
    )

    with pytest.raises(RuntimeError) as caught:
        tune.run_tuning()

    assert_plain_error(caught, seen)


@pytest.mark.parametrize("kind", KINDS)
def test_fallback_non_save_error_propagates(monkeypatch, tmp_path, kind):
    from taxi_duration import tune

    seen = setup_run(
        monkeypatch,
        tmp_path,
        trials=[DEEP_A, DEEP_B],
        size_for=deep_is_huge,
        scores_for=scores,
    )
    first = tune.fallback_attempts(DEEP_B)[0]
    assert (first["max_depth"], first["n_estimators"]) == (15, 60)
    control_model_dir(
        monkeypatch, {(15, 60): {kind: RuntimeError("model file is corrupt")}}
    )

    with pytest.raises(RuntimeError) as caught:
        tune.run_tuning()

    assert_plain_error(caught, seen)
