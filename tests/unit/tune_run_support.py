"""Stubs shared by the run_tuning tests: no data, no server, no real fits.

Seams patched (all module attributes, so run_tuning sees them as globals):
tune._fit, tune.measure_baseline_ms, tune.measure_predict_ms, tune.score_params,
tune.suggest_params, tune.run_study, tune.prepare_model_dir (recorder),
artifacts.save_model, artifacts.load_model, artifacts.estimate_size (FakeFitted has
no trees; estimate_for gives the size estimate, default small), splits.*, train.read_lock,
train._experiment_id and mlflow.MlflowClient.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HUGE = 400_000_000  # over the 300 MB limit
SMALL = 1_000_000


def make_frame(n=300):
    rng = np.random.default_rng(1)
    distance = rng.uniform(0.5, 12, n)
    return pd.DataFrame(
        {
            "trip_id": [f"t{i:04d}" for i in range(n)],
            "pickup_at": pd.Timestamp("2025-01-05")
            + pd.to_timedelta(np.arange(n) * 30, unit="m"),
            "pickup_zone_id": rng.choice([10, 20], n),
            "dropoff_zone_id": rng.choice([30, 40], n),
            "trip_distance": distance,
            "duration_minutes": 2 + 4 * distance + rng.normal(0, 0.3, n),
        }
    )


def make_folds(df):
    ids = df["trip_id"].tolist()
    return [
        {"train": ids[:120], "valid": ids[120:170]},
        {"train": ids[:170], "valid": ids[170:220]},
        {"train": ids[:220], "valid": ids[220:270]},
    ]


def params(depth, trees=50):
    return {
        "n_estimators": trees,
        "max_depth": depth,
        "min_samples_leaf": 2,
        "max_features": 0.5,
        "max_samples": 0.8,
    }


class FakeTrial:
    def __init__(self, number):
        self.number = number


class FakeFitted:
    def __init__(self, fit_params):
        self.params = fit_params
        self.named_steps = {"model": self}

    def set_params(self, **kwargs):
        return self

    def predict(self, rows):
        return np.zeros(len(rows))


class FakeClient:
    """Counts MLflow runs. Has no registry methods on purpose."""

    def __init__(self):
        self.runs = []

    def create_run(self, experiment_id):
        self.runs.append(experiment_id)
        return SimpleNamespace(info=SimpleNamespace(run_id=f"run{len(self.runs)}"))

    def set_tag(self, *args, **kwargs):
        pass

    log_param = log_metric = set_terminated = set_tag


class Seen:
    """What the stubs saw."""

    def __init__(self):
        self.client = FakeClient()
        self.fits = []
        self.baseline_timed = []
        self.scored = []
        self.saves = []
        self.prepares = []
        self.unprepared_saves = []
        self.experiment_calls = []
        self.results = None
        self.prepared = None


def setup_run(
    monkeypatch,
    tmp_path,
    *,
    trials=(),
    size_for=lambda p: SMALL,
    ms_for=lambda p: 1.0,
    scores_for=lambda p: [3.0, 3.0, 3.0],
    estimate_for=lambda p: SMALL,
    edit_lock=None,
):
    import mlflow

    from taxi_duration import artifacts, config, splits, train, tune

    seen = Seen()
    df = make_frame()
    folds = make_folds(df)
    lock = {
        "fold_scores": [4.0, 3.9, 3.8],
        "sample_sha256": train.fingerprint(df["trip_id"]),
        "folds": train.fold_fingerprints(folds),
    }
    if edit_lock:
        edit_lock(lock)
    seen.results = tmp_path / "results"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    monkeypatch.setenv("MODEL_TRUSTED_TYPES", "numpy.ndarray")
    monkeypatch.setattr(mlflow, "MlflowClient", lambda *a, **k: seen.client)

    def experiment_id(client, uri):
        seen.experiment_calls.append(uri)
        return "exp"

    monkeypatch.setattr(train, "_experiment_id", experiment_id)
    monkeypatch.setattr(config, "RESULTS_DIR", str(seen.results))
    monkeypatch.setattr(tune, "MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setattr(splits, "load_train", lambda: df)
    monkeypatch.setattr(splits, "draw_sample", lambda frame: frame)
    monkeypatch.setattr(splits, "make_folds", lambda frame: folds)
    monkeypatch.setattr(train, "read_lock", lambda *a, **k: lock)

    def boom(*args, **kwargs):
        raise AssertionError("nothing may be registered by run_tuning")

    monkeypatch.setattr(tune, "register_champion", boom)

    def baseline_ms(sample, fold_list):
        seen.baseline_timed.append(1)
        return 1.0

    monkeypatch.setattr(tune, "measure_baseline_ms", baseline_ms)

    def fit(rows, fit_params):
        seen.fits.append(dict(fit_params))
        return FakeFitted(dict(fit_params))

    monkeypatch.setattr(tune, "_fit", fit)

    def score(sample, score_params, **kwargs):
        seen.scored.append(dict(score_params))
        return list(scores_for(score_params))

    monkeypatch.setattr(tune, "score_params", score)
    monkeypatch.setattr(tune, "measure_predict_ms", lambda f, row: ms_for(f.params))
    monkeypatch.setattr(tune, "suggest_params", lambda t: dict(trials[t.number]))

    def run_study(study, objective, **kwargs):
        for number in range(len(trials)):
            objective(FakeTrial(number))

    monkeypatch.setattr(tune, "run_study", run_study)

    stored = {}

    def prepare(path):
        seen.prepares.append(str(path))
        seen.prepared = str(path)
        Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(tune, "prepare_model_dir", prepare, raising=False)

    def save(fitted, folder):
        key = str(folder)
        seen.saves.append(key)
        if seen.prepared != key:
            seen.unprepared_saves.append(key)
        seen.prepared = None
        stored[key] = fitted
        return size_for(fitted.params)

    monkeypatch.setattr(artifacts, "save_model", save)
    monkeypatch.setattr(
        artifacts, "load_model", lambda folder, trusted: stored[str(folder)]
    )
    monkeypatch.setattr(artifacts, "estimate_size", lambda f: estimate_for(f.params))
    return seen


def read_trials(seen):
    return json.loads((seen.results / "trials.json").read_text())
