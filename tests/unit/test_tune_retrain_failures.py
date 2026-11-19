"""Failed retrain record (review 4, item 3).

Interface tune.run_tuning must provide: when first_passing_after_retrain returns None,
trials.json holds "final": null and "retrain_failures": a list, in the order tried, of
{"name", "params", "size_bytes", "predict_ms", "reason"} with exactly these keys, one
per tried candidate. name and params are the candidate's. size_bytes and predict_ms are
the measured values, None when not measured. reason is one of "disk_error",
"size_limit", "speed_limit", "size_and_speed_limit". The NeedsApproval message names the
number of failures as a whole number and contains "disk" (any case) if and only if at
least one reason is "disk_error".
"""

import re

import pytest
from tune_retrain_support import control_model_dir
from tune_run_support import HUGE, SMALL, params, read_trials, setup_run

ONE = params(depth=12, trees=50)
TWO = params(depth=10, trees=60)
THREE = params(depth=9, trees=70)
FOUR = params(depth=8, trees=80)
SCORES = {(12, 50): 3.0, (10, 60): 3.2, (9, 70): 3.3, (8, 80): 3.4}
SLOW = 100.0
KEYS = {"name", "params", "size_bytes", "predict_ms", "reason"}


def scores(p):
    return [SCORES[(p["max_depth"], p["n_estimators"])]] * 3


def run(monkeypatch, tmp_path, trials, outcomes):
    from taxi_duration import tune

    seen = setup_run(monkeypatch, tmp_path, trials=trials, scores_for=scores)
    control_model_dir(monkeypatch, outcomes)
    with pytest.raises(tune.NeedsApproval) as caught:
        tune.run_tuning()
    return seen, str(caught.value)


def test_each_reason_is_recorded_in_order(monkeypatch, tmp_path):
    seen, message = run(
        monkeypatch,
        tmp_path,
        [ONE, TWO, THREE, FOUR],
        {
            (12, 50): {"size": HUGE, "ms": 1.0},
            (10, 60): {"size": SMALL, "ms": SLOW},
            (9, 70): {"disk": True},
            (8, 80): {"size": HUGE, "ms": SLOW},
        },
    )

    data = read_trials(seen)
    assert data["final"] is None
    failures = data["retrain_failures"]
    assert all(set(f) == KEYS for f in failures)
    assert [f["name"] for f in failures] == ["trial_0", "trial_1", "trial_2", "trial_3"]
    assert [f["params"] for f in failures] == [ONE, TWO, THREE, FOUR]
    assert [f["reason"] for f in failures] == [
        "size_limit",
        "speed_limit",
        "disk_error",
        "size_and_speed_limit",
    ]
    assert (failures[0]["size_bytes"], failures[0]["predict_ms"]) == (HUGE, 1.0)
    assert (failures[1]["size_bytes"], failures[1]["predict_ms"]) == (SMALL, SLOW)
    assert failures[2]["predict_ms"] is None
    assert failures[2]["size_bytes"] in (None, SMALL)  # unmeasured or the estimate
    assert (failures[3]["size_bytes"], failures[3]["predict_ms"]) == (HUGE, SLOW)
    assert re.search(r"\b4\b", message)
    assert "disk" in message.lower()


def test_no_disk_error_means_no_disk_in_the_message(monkeypatch, tmp_path):
    seen, message = run(
        monkeypatch,
        tmp_path,
        [ONE, TWO],
        {(12, 50): {"size": HUGE, "ms": 1.0}, (10, 60): {"size": SMALL, "ms": SLOW}},
    )

    failures = read_trials(seen)["retrain_failures"]
    assert [f["reason"] for f in failures] == ["size_limit", "speed_limit"]
    assert re.search(r"\b2\b", message)
    assert "disk" not in message.lower()


def test_a_single_disk_error_is_named(monkeypatch, tmp_path):
    seen, message = run(monkeypatch, tmp_path, [ONE], {(12, 50): {"disk": True}})

    data = read_trials(seen)
    assert data["final"] is None
    assert [f["reason"] for f in data["retrain_failures"]] == ["disk_error"]
    assert re.search(r"\b1\b", message)
    assert "disk" in message.lower()
