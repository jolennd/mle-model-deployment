"""Decision rule, MLflow URI check and model folder cleaning (batch 4).

Interface tune.py must provide:
- choose_final(baseline, candidates): in order. 1) among candidates that pass both
  limits, are at least 2 percent better than the baseline's mean and win every fold
  against the baseline's fold scores: the best mean (the baseline's own limits do
  not matter). 2) else the baseline if it passes both limits. 3) else the best mean
  among candidates that pass both limits. 4) else None.
- _check_uri(uri) -> uri: accepts only scheme http, hostname 127.0.0.1 (urllib.parse),
  with an optional port and path. Anything else raises ValueError.
  register_champion(..., tracking_uri=...) applies it only to the URI taken from the
  environment, so an explicit local sqlite store still works in tests.
- prepare_model_dir(path): removes all content under path and recreates it empty;
  creates it (with parents) if missing.
"""

import pytest

from taxi_duration import config

OK = 1_000_000
SLOW = config.MAX_PREDICT_MS


def rec(name, folds, size=OK, ms=1.0):
    return {"name": name, "fold_rmse": folds, "size_bytes": size, "predict_ms": ms}


def flat(name, mean, **kwargs):
    return rec(name, [mean, mean, mean], **kwargs)


FAILING_BASELINE = flat("baseline", 100.0, ms=SLOW)
PASSING_BASELINE = flat("baseline", 100.0)


def test_eligible_candidate_beats_a_lower_mean_that_is_not_eligible():
    from taxi_duration import tune

    eligible = flat("eligible", 90.0)
    loses_a_fold = rec("loses_fold", [70.0, 70.0, 101.0])  # mean 80.3, fold 3 worse
    under_two_percent = flat("under_2pct", 98.5)
    for baseline in (FAILING_BASELINE, PASSING_BASELINE):
        got = tune.choose_final(baseline, [loses_a_fold, under_two_percent, eligible])
        assert got["name"] == "eligible"


def test_best_eligible_mean_wins_among_eligible():
    from taxi_duration import tune

    candidates = [flat("a", 92.0), flat("b", 88.0), flat("c", 95.0)]
    assert tune.choose_final(FAILING_BASELINE, candidates)["name"] == "b"


def test_eligible_candidate_that_fails_a_limit_is_skipped():
    from taxi_duration import tune

    too_slow = flat("best_but_slow", 80.0, ms=SLOW)
    eligible = flat("eligible", 90.0)
    got = tune.choose_final(FAILING_BASELINE, [too_slow, eligible])
    assert got["name"] == "eligible"


def test_step_three_best_mean_when_nothing_is_eligible():
    from taxi_duration import tune

    loses_a_fold = rec("loses_fold", [70.0, 70.0, 101.0])
    under_two_percent = flat("under_2pct", 98.5)
    big = flat("big", 50.0, size=config.MODEL_SIZE_LIMIT_BYTES)
    got = tune.choose_final(FAILING_BASELINE, [under_two_percent, loses_a_fold, big])
    assert got["name"] == "loses_fold"


def test_step_two_keeps_a_passing_baseline_and_none_when_all_fail():
    from taxi_duration import tune

    loses_a_fold = rec("loses_fold", [70.0, 70.0, 101.0])
    assert tune.choose_final(PASSING_BASELINE, [loses_a_fold])["name"] == "baseline"
    slow = flat("slow", 80.0, ms=SLOW)
    assert tune.choose_final(FAILING_BASELINE, [slow]) is None


@pytest.mark.parametrize(
    "uri",
    [
        "http://127.0.0.1",
        "http://127.0.0.1/",
        "http://127.0.0.1:5001",
        "http://127.0.0.1:5001/some/path",
    ],
)
def test_check_uri_accepts_the_loopback_server(uri):
    from taxi_duration import tune

    assert tune._check_uri(uri) == uri


@pytest.mark.parametrize(
    "uri",
    [
        "sqlite:///x.db",
        "http://127.0.0.1.evil.com",
        "http://127.0.0.1.evil.com:5001/x",
        "http://localhost:5001",
        "https://127.0.0.1",
        "http://127.0.0.1@evil.com",
        "",
    ],
)
def test_check_uri_refuses_everything_else(uri):
    from taxi_duration import tune

    with pytest.raises(ValueError):
        tune._check_uri(uri)


def test_prepare_model_dir_empties_or_creates_the_folder(tmp_path):
    from taxi_duration import tune

    folder = tmp_path / "models" / "candidate"
    (folder / "sub" / "deeper").mkdir(parents=True)
    (folder / "model.skops").write_bytes(b"old")
    (folder / "sub" / "deeper" / "x.txt").write_text("old")
    keep = tmp_path / "models" / "other.txt"
    keep.write_text("not mine")

    tune.prepare_model_dir(folder)

    assert folder.is_dir()
    assert list(folder.iterdir()) == []
    assert keep.read_text() == "not mine"

    missing = tmp_path / "a" / "b" / "model"
    tune.prepare_model_dir(str(missing))
    assert missing.is_dir()
    assert list(missing.iterdir()) == []


def test_fallback_of_the_baseline_uses_the_scikit_learn_defaults():
    from taxi_duration import tune

    assert tune.skipped_caps({}) == []
    attempts = tune.fallback_attempts({})
    caps = [a["max_depth"] for a in attempts[:6]]
    assert caps == [25, 20, 15, 12, 10, 8]
    assert all(a["n_estimators"] == 100 for a in attempts[:6])
    assert [(a["max_depth"], a["n_estimators"]) for a in attempts[6:]] == [
        (8, 50),
        (8, 25),
    ]
    assert len(attempts) == 8
