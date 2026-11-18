"""Skipped depth caps in the fallback (ADR 8 point 13).

Interface tune.py must provide:
- skipped_caps(params) -> list[int]: caps of 25, 20, 15, 12, 10, 8 that are at or
  above params["max_depth"], in plan order. max_depth None skips none.
- fallback_attempts(params) omits the skipped caps; halving steps stay at depth 8
  (or the candidate's own depth when all caps are skipped, which is 8 or less).
- select_fallback(params, check) -> (params, misses): with no attempts, check is
  called once on the input params and misses is True when it fails a limit.
"""

from taxi_duration import config

BASE = {
    "n_estimators": 300,
    "max_depth": 30,
    "min_samples_leaf": 3,
    "max_features": 0.5,
    "max_samples": 0.8,
}

OK = (1_000_000, 1.0)
BAD = (config.MODEL_SIZE_LIMIT_BYTES, 1.0)


def with_depth(depth, trees=300):
    return {**BASE, "max_depth": depth, "n_estimators": trees}


def test_skipped_caps_by_candidate_depth():
    from taxi_duration import tune

    before = with_depth(12)
    assert tune.skipped_caps(with_depth(30)) == []
    assert tune.skipped_caps(before) == [25, 20, 15, 12]
    assert tune.skipped_caps(with_depth(8)) == [25, 20, 15, 12, 10, 8]
    assert tune.skipped_caps(with_depth(None)) == []
    assert before == with_depth(12)  # input untouched


def test_fallback_skips_caps_at_or_above_own_depth():
    from taxi_duration import tune

    got = tune.fallback_attempts(with_depth(12))
    assert [a["max_depth"] for a in got] == [10, 8, 8, 8, 8, 8]
    assert [a["n_estimators"] for a in got] == [300, 300, 150, 75, 37, 25]

    got = tune.fallback_attempts(with_depth(8))
    assert [a["max_depth"] for a in got] == [8, 8, 8, 8]
    assert [a["n_estimators"] for a in got] == [150, 75, 37, 25]

    got = tune.fallback_attempts(with_depth(None))
    assert [a["max_depth"] for a in got] == [25, 20, 15, 12, 10, 8, 8, 8, 8, 8]

    assert len(tune.fallback_attempts(with_depth(30))) == 10  # nothing skipped


def test_fallback_skips_all_caps_and_halving_empty_returns_input():
    from taxi_duration import tune

    params = with_depth(8, trees=25)
    assert tune.fallback_attempts(params) == []

    calls = []

    def fails(p):
        calls.append(p)
        return BAD

    chosen, misses = tune.select_fallback(params, fails)
    assert chosen == params
    assert misses is True
    assert calls == [params]

    chosen, misses = tune.select_fallback(params, lambda p: OK)
    assert chosen == params
    assert misses is False


def test_fallback_skips_still_end_on_last_attempt_with_misses():
    from taxi_duration import tune

    seen = []

    def never(p):
        seen.append(p)
        return BAD

    chosen, misses = tune.select_fallback(with_depth(12), never)
    assert (chosen["max_depth"], chosen["n_estimators"]) == (8, 25)
    assert misses is True
    assert [p["max_depth"] for p in seen][:2] == [10, 8]  # no cap 25 to 12
    assert len(seen) == 6
