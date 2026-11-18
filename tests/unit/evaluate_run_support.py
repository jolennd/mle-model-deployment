"""Stubs shared by the run_evaluation tests: no data, no server, no real fits.

Seams patched (module attributes): splits.load_train, splits.draw_sample,
splits.make_folds, splits.load_test, train.read_lock, evaluate.cv_reports,
evaluate.refit_baseline, verify_model.verify_then_load and mlflow.MlflowClient.
Every call is appended to seen.events, in call order.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

GOOD_URI = "http://127.0.0.1:5001"
MODEL_URI = "models:/taxi-duration@champion"
SHA = "ab" * 32
FINAL_PARAMS = {"n_estimators": 5, "max_depth": 8}


def make_test_frame():
    pickup = pd.date_range("2025-01-25 08:00", periods=6, freq="h")
    return pd.DataFrame(
        {
            "trip_id": [f"t{i}" for i in range(6)],
            "pickup_at": pickup,
            "dropoff_at": pickup + pd.Timedelta(minutes=15),
            "pickup_zone_id": [132, 10, 10, 20, 20, 138],
            "dropoff_zone_id": [30, 30, 40, 40, 30, 40],
            "trip_distance": [10.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "duration_minutes": [30.0, 5.0, 10.0, 15.0, 20.0, 40.0],
        }
    )


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


class Seen:
    def __init__(self):
        self.events = []
        self.verify_kwargs = []
        self.results = None

    def count(self, name):
        return self.events.count(name)


def setup_evaluation(
    monkeypatch,
    tmp_path,
    *,
    tracking_uri=GOOD_URI,
    model_uri=MODEL_URI,
    trials=FINAL_PARAMS,
    verify_error=None,
):
    """trials: params written to trials.json["final"]["params"], or None for no file."""
    import mlflow

    from taxi_duration import evaluate, splits, train, verify_model

    seen = Seen()
    seen.results = tmp_path / "results"
    seen.results.mkdir()
    if trials is not None:
        (seen.results / "trials.json").write_text(
            json.dumps({"final": {"params": trials}})
        )

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MODEL_URI", model_uri)
    monkeypatch.setenv("MODEL_SHA256", SHA)
    monkeypatch.setenv("MODEL_TRUSTED_TYPES", "sklearn.tree._tree.Tree")

    sample = pd.DataFrame({"trip_id": ["a", "b"], "duration_minutes": [10.0, 20.0]})

    def load_train():
        seen.events.append("load_train")
        return "train-rows"

    monkeypatch.setattr(splits, "load_train", load_train)
    monkeypatch.setattr(splits, "draw_sample", lambda frame, *a, **k: sample)
    monkeypatch.setattr(splits, "make_folds", lambda frame: [])
    monkeypatch.setattr(train, "read_lock", lambda *a, **k: {"sample_sha256": "x"})

    def load_test():
        seen.events.append("load_test")
        return make_test_frame()

    monkeypatch.setattr(splits, "load_test", load_test)

    def cv_reports(*args, **kwargs):
        seen.events.append("cv_reports")
        return {}

    monkeypatch.setattr(evaluate, "cv_reports", cv_reports)

    def refit_baseline(*args, **kwargs):
        seen.events.append("refit_baseline")
        return Constant(10.0)

    monkeypatch.setattr(evaluate, "refit_baseline", refit_baseline)

    def verify_then_load(uri, **kwargs):
        seen.events.append("verify")
        seen.verify_kwargs.append((uri, kwargs))
        if verify_error is not None:
            raise verify_error
        return Constant(20.0), SHA

    monkeypatch.setattr(verify_model, "verify_then_load", verify_then_load)

    lookups = []

    def get_model_version_by_alias(name, alias):
        seen.events.append("alias_lookup")
        lookups.append(1)
        return SimpleNamespace(version=str(2 + len(lookups)))  # "3", then "4", ...

    client = SimpleNamespace(get_model_version_by_alias=get_model_version_by_alias)
    monkeypatch.setattr(mlflow, "MlflowClient", lambda *a, **k: client)
    return seen
