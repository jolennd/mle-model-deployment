"""Tuning follows docs/tuning-plan.md (batch 1).

Interface tune.py must provide:
- suggest_params(trial) -> dict: draws the search space from an Optuna-style trial
  (suggest_int / suggest_float / suggest_categorical).
- MAX_TRIALS, MAX_SECONDS, make_study(), run_study(study, objective, n_trials, timeout).
- score_params(sample, params) -> list of RMSE per fold, on splits.make_folds(sample),
  through train.cross_validate.
- choose(baseline, candidates) -> the chosen record (decision rule steps 1 and 2).
- MAX_PREDICT_MS (used in batch 2).

Candidate record: {"name": str, "fold_rmse": [3 floats], "size_bytes": int,
"predict_ms": float}.
"""

import time
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd

from taxi_duration import splits, train

# Written out from docs/tuning-plan.md, not imported from tune.
PLAN_INTS = {
    "n_estimators": (50, 300, False),
    "max_depth": (8, 30, False),
    "min_samples_leaf": (1, 50, True),
}
PLAN_FLOATS = {"max_samples": (0.3, 1.0, False)}
PLAN_CHOICES = {"max_features": ["sqrt", 0.3, 0.5, 1.0]}


class RecordingTrial:
    """Stands in for an Optuna trial and records every draw."""

    def __init__(self):
        self.ints, self.floats, self.choices = {}, {}, {}

    def suggest_int(self, name, low, high, step=1, log=False):
        self.ints[name] = (low, high, log)
        return low

    def suggest_float(self, name, low, high, step=None, log=False):
        self.floats[name] = (low, high, log)
        return low

    def suggest_categorical(self, name, choices):
        self.choices[name] = list(choices)
        return choices[0]


def test_search_space_matches_plan():
    from taxi_duration import tune

    trial = RecordingTrial()
    params = tune.suggest_params(trial)

    assert trial.ints == PLAN_INTS
    assert trial.floats == PLAN_FLOATS
    assert trial.choices == PLAN_CHOICES
    assert set(params) == set(PLAN_INTS) | set(PLAN_FLOATS) | set(PLAN_CHOICES)


def _first_params(study, n=3):
    study.optimize(
        lambda t: t.suggest_float("x", 0, 1) + t.suggest_int("k", 1, 50, log=True),
        n_trials=n,
    )
    return [t.params for t in study.trials]


def test_budget_and_seed_match_plan():
    from taxi_duration import tune

    assert tune.MAX_TRIALS == 30
    assert tune.MAX_SECONDS == 45 * 60 == 2700

    study = tune.make_study()
    assert study.direction == optuna.study.StudyDirection.MINIMIZE
    assert isinstance(study.sampler, optuna.samplers.TPESampler)
    assert isinstance(study.pruner, optuna.pruners.NopPruner)
    reference = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    assert _first_params(study) == _first_params(reference)

    # run_study passes the plan budget to the study by default
    seen = {}
    spy = SimpleNamespace(optimize=lambda fn, **kw: seen.update(kw))
    tune.run_study(spy, lambda t: 0.0)
    assert seen["n_trials"] == 30
    assert seen["timeout"] == 2700


def test_trial_at_time_limit_finishes_and_no_trial_starts_after():
    from taxi_duration import tune

    study = optuna.create_study()

    def slow(trial):
        trial.suggest_float("x", 0, 1)
        time.sleep(0.15)
        return 1.0

    tune.run_study(study, slow, n_trials=30, timeout=0.05)

    assert len(study.trials) == 1
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE


def make_frame():
    """Three trips per day for January, with a linear duration."""
    rng = np.random.default_rng(1)
    rows = []
    for n, (day, offset, minutes) in enumerate(
        (d, o, m)
        for d in pd.date_range("2025-01-01", "2025-01-31", freq="D")
        for o, m in ((2 * 60, 10), (12 * 60, 10), (23 * 60 + 50, 30))
    ):
        pickup = day + pd.Timedelta(minutes=offset)
        distance = float(rng.uniform(0.5, 12))
        rows.append(
            {
                "trip_id": f"t{n:04d}",
                "pickup_at": pickup,
                "dropoff_at": pickup + pd.Timedelta(minutes=minutes),
                "pickup_zone_id": int(rng.choice([10, 20])),
                "dropoff_zone_id": int(rng.choice([30, 40])),
                "trip_distance": distance,
                "duration_minutes": 2 + 4 * distance + float(rng.normal(0, 0.3)),
            }
        )
    return pd.DataFrame(rows)


SMALL_PARAMS = {
    "n_estimators": 5,
    "max_depth": 8,
    "min_samples_leaf": 2,
    "max_features": 0.5,
    "max_samples": 0.7,
}


def test_trials_use_baseline_folds(monkeypatch):
    from taxi_duration import tune

    sample = make_frame()
    baseline_folds = splits.make_folds(sample)
    real = train.cross_validate
    seen = {}

    def spy(df, folds, pipeline_factory):
        seen["folds"] = folds
        seen["settings"] = pipeline_factory().get_params()
        return real(df, folds, pipeline_factory=pipeline_factory)

    monkeypatch.setattr(train, "cross_validate", spy)

    scores = tune.score_params(sample, SMALL_PARAMS)

    assert seen["folds"] == baseline_folds  # same fold ids as the baseline
    assert len(scores) == 3
    assert all(isinstance(s, float) and s > 0 for s in scores)
    settings = seen["settings"]
    for name, value in SMALL_PARAMS.items():
        assert settings[f"model__{name}"] == value
    assert settings["model__random_state"] == 42


def record(name, fold_rmse):
    return {
        "name": name,
        "fold_rmse": fold_rmse,
        "size_bytes": 1_000_000,
        "predict_ms": 1.0,
    }


BASELINE = record("baseline", [100.0, 100.0, 100.0])


def test_baseline_kept_when_gain_under_threshold():
    from taxi_duration import tune

    just_under = record("just_under", [98.01, 98.01, 98.01])  # 1.99 percent lower
    assert tune.choose(BASELINE, [just_under])["name"] == "baseline"
    assert tune.choose(BASELINE, [])["name"] == "baseline"

    exactly = record("exactly_2_percent", [98.0, 98.0, 98.0])  # 2 percent lower
    assert tune.choose(BASELINE, [exactly])["name"] == "exactly_2_percent"


def test_candidate_must_win_every_fold():
    from taxi_duration import tune

    best_mean_loses_a_fold = record("loses_fold_3", [80.0, 80.0, 100.5])
    ok = record("ok", [95.0, 95.0, 95.0])
    best_eligible = record("best_eligible", [90.0, 90.0, 90.0])

    assert tune.choose(BASELINE, [best_mean_loses_a_fold])["name"] == "baseline"
    chosen = tune.choose(BASELINE, [best_mean_loses_a_fold, ok, best_eligible])
    assert chosen["name"] == "best_eligible"
    tie_fold = record("ties_a_fold", [80.0, 80.0, 100.0])  # equal is not a win
    assert tune.choose(BASELINE, [tie_fold])["name"] == "baseline"
