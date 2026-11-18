"""Limits, final choice and fallback of docs/tuning-plan.md (batch 2).

Interface tune.py must provide (config holds the limits, not tune):
- passes_limits(size_bytes, predict_ms) -> bool: size < config.MODEL_SIZE_LIMIT_BYTES
  and predict_ms < config.MAX_PREDICT_MS, both strict.
- measure_predict_ms(fitted, row) -> float: sets fitted.named_steps["model"] to
  n_jobs=1 through set_params(n_jobs=1), makes 1 warm-up call fitted.predict(row),
  then 100 timed calls with time.perf_counter() (module attribute access, not a
  from-import), returns the median in milliseconds.
- choose_final(baseline, candidates) -> record or None: decision steps 1 to 3.
  None when nothing passes both limits (the caller then uses the fallback).
- fallback_attempts(params) -> list of param dicts, in order.
- select_fallback(params, check) -> (params, misses).
- first_passing_after_retrain(ordered_candidates, retrain_check) -> record or None.
"""

import time

import pytest

from taxi_duration import config


def test_size_and_speed_limits_enforced():
    from taxi_duration import tune

    size = config.MODEL_SIZE_LIMIT_BYTES
    ms = config.MAX_PREDICT_MS
    assert tune.passes_limits(size - 1, ms - 0.001) is True
    assert tune.passes_limits(size, ms - 0.001) is False  # size equal fails
    assert tune.passes_limits(size + 1, 1.0) is False
    assert tune.passes_limits(size - 1, ms) is False  # time equal fails
    assert tune.passes_limits(1_000, ms + 1) is False
    assert tune.passes_limits(size, ms) is False


class FakeForest:
    def __init__(self):
        self.n_jobs = -1

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self


class TimedModel:
    """Counts calls, records n_jobs at call time, advances a fake clock."""

    def __init__(self, durations_ms, clock):
        self.forest = FakeForest()
        self.named_steps = {"model": self.forest}
        self.durations_ms = list(durations_ms)
        self.clock = clock
        self.n_jobs_seen = []
        self.rows = []

    def predict(self, row):
        self.n_jobs_seen.append(self.forest.n_jobs)
        self.rows.append(row)
        self.clock[0] += self.durations_ms[len(self.rows) - 1] / 1000
        return [0.0]


def test_measure_predict_ms_warms_up_and_returns_median(monkeypatch):
    from taxi_duration import tune

    clock = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    # warm-up (ignored), then 49 slow and 51 fast calls: median 2.0, mean 50.02
    durations = [1000.0] + [100.0] * 49 + [2.0] * 51
    model = TimedModel(durations, clock)
    row = object()

    got = tune.measure_predict_ms(model, row)

    assert len(model.rows) == 101  # 1 warm-up + 100 timed
    assert all(r is row for r in model.rows)
    assert model.n_jobs_seen == [1] * 101  # one thread from the first call on
    assert got == pytest.approx(2.0)


BASE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 30,
    "min_samples_leaf": 3,
    "max_features": 0.5,
    "max_samples": 0.8,
}


def rec(name, mean, size=1_000_000, ms=1.0):
    return {
        "name": name,
        "fold_rmse": [mean, mean, mean],
        "size_bytes": size,
        "predict_ms": ms,
    }


def test_fallback_when_no_candidate_passes():
    from taxi_duration import tune

    # Step 3: baseline fails a limit, best mean CV RMSE among passing candidates.
    baseline = rec("baseline", 100.0, ms=60.0)
    a = rec("a", 95.0)
    too_big = rec("too_big", 90.0, size=config.MODEL_SIZE_LIMIT_BYTES)
    slow = rec("slow", 91.0, ms=config.MAX_PREDICT_MS)
    c = rec("c", 92.0)
    assert tune.choose_final(baseline, [a, too_big, slow, c])["name"] == "c"
    # Step 3 needs no 2 percent gain
    assert tune.choose_final(baseline, [rec("flat", 100.0)])["name"] == "flat"
    # Steps 1 and 2 still apply when the baseline passes
    ok_base = rec("baseline", 100.0)
    assert tune.choose_final(ok_base, [rec("better", 90.0)])["name"] == "better"
    assert tune.choose_final(ok_base, [rec("tiny", 99.5)])["name"] == "baseline"
    # An eligible-on-CV candidate that fails a limit is not chosen
    assert tune.choose_final(ok_base, [rec("fast_big", 80.0, ms=99.0)])["name"] == (
        "baseline"
    )
    # Nothing passes: None, the caller uses the fallback
    assert tune.choose_final(baseline, [too_big, slow]) is None


def test_fallback_attempts_depth_caps_then_halving():
    from taxi_duration import tune

    before = dict(BASE_PARAMS)
    attempts = tune.fallback_attempts(BASE_PARAMS)
    assert BASE_PARAMS == before  # input untouched

    depths = [a["max_depth"] for a in attempts]
    trees = [a["n_estimators"] for a in attempts]
    assert depths == [25, 20, 15, 12, 10, 8, 8, 8, 8, 8]
    assert trees == [300] * 6 + [150, 75, 37, 25]
    for a in attempts:  # other settings are kept
        assert a["min_samples_leaf"] == 3
        assert a["max_features"] == 0.5
        assert a["max_samples"] == 0.8

    def halving(n):
        got = tune.fallback_attempts({**BASE_PARAMS, "n_estimators": n})
        return [a["n_estimators"] for a in got[6:]]

    assert halving(101) == [50, 25]
    assert halving(60) == [30, 25]  # half of 30 is 15, below 25, so 25
    assert halving(30) == [25]
    assert halving(25) == []  # already at the minimum
    last = tune.fallback_attempts({**BASE_PARAMS, "n_estimators": 25})[-1]
    assert (last["max_depth"], last["n_estimators"]) == (8, 25)


def test_select_fallback_first_pass_or_last_with_misses():
    from taxi_duration import tune

    ok = (1_000_000, 1.0)
    bad = (config.MODEL_SIZE_LIMIT_BYTES, 1.0)

    calls = []

    def passes_at_15(params):
        calls.append(params)
        return ok if params["max_depth"] <= 15 else bad

    chosen, misses = tune.select_fallback(BASE_PARAMS, passes_at_15)
    assert (chosen["max_depth"], chosen["n_estimators"]) == (15, 300)
    assert misses is False
    assert [p["max_depth"] for p in calls] == [25, 20, 15]  # stops at first pass

    def passes_on_speed_after_halving(params):
        return (1, 1.0) if params["n_estimators"] <= 75 else (1, 80.0)

    chosen, misses = tune.select_fallback(BASE_PARAMS, passes_on_speed_after_halving)
    assert (chosen["max_depth"], chosen["n_estimators"]) == (8, 75)
    assert misses is False

    seen = []

    def never(params):
        seen.append(params)
        return bad

    chosen, misses = tune.select_fallback(BASE_PARAMS, never)
    assert (chosen["max_depth"], chosen["n_estimators"]) == (8, 25)
    assert misses is True
    assert len(seen) == len(tune.fallback_attempts(BASE_PARAMS))


def test_retrain_failure_moves_to_next_best_passing_candidate():
    from taxi_duration import tune

    ordered = [rec("first", 90.0), rec("second", 91.0), rec("third", 92.0)]
    results = {
        "first": (config.MODEL_SIZE_LIMIT_BYTES + 1, 1.0),
        "second": (1_000, config.MAX_PREDICT_MS),
        "third": (1_000, 10.0),
    }
    tried = []

    def retrain_check(candidate):
        tried.append(candidate["name"])
        return results[candidate["name"]]

    got = tune.first_passing_after_retrain(ordered, retrain_check)
    assert got["name"] == "third"
    assert tried == ["first", "second", "third"]

    tried.clear()
    results["first"] = (1_000, 1.0)
    assert tune.first_passing_after_retrain(ordered, retrain_check)["name"] == "first"
    assert tried == ["first"]  # no retrain after a pass

    assert tune.first_passing_after_retrain(ordered, lambda c: (10**9, 1.0)) is None
