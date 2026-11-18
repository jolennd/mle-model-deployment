"""Baseline test-week score from an in-memory refit (ADR 8 points 15 and 16).

Interface to implement in evaluate.py:
- refit_baseline(sample, lock, *, pipeline_factory=None) -> fitted pipeline.
  Checks train.fingerprint(sample["trip_id"]) == lock["sample_sha256"] first and
  raises ValueError naming sample_sha256 before any fitting. Fits on the whole
  sample (features.build_features, duration_minutes). Default factory is
  features.make_pipeline (RandomForest defaults, random_state 42). Writes nothing.
- run_evaluation no longer reads BASELINE_MODEL_URI or BASELINE_MODEL_SHA256. Its
  baseline loader calls refit_baseline(draw_sample(load_train()), read_lock()).
"""

import inspect

import numpy as np
import pandas as pd
import pytest


def make_sample(n=240):
    rng = np.random.default_rng(3)
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


def test_baseline_refit_checks_sample_hash(tmp_path, monkeypatch):
    from taxi_duration import artifacts, evaluate, features, train

    def boom(*args, **kwargs):
        raise AssertionError("the refit must not save or load a model file")

    monkeypatch.setattr(artifacts, "save_model", boom)
    monkeypatch.setattr(artifacts, "load_model", boom)
    monkeypatch.chdir(tmp_path)

    sample = make_sample()
    calls = []

    def factory():
        calls.append(1)
        return features.make_pipeline().set_params(model__n_estimators=5)

    bad_lock = {"sample_sha256": "0" * 64}
    with pytest.raises(ValueError, match="sample_sha256"):
        evaluate.refit_baseline(sample, bad_lock, pipeline_factory=factory)
    assert calls == []

    lock = {"sample_sha256": train.fingerprint(sample["trip_id"])}
    fitted = evaluate.refit_baseline(sample, lock, pipeline_factory=factory)
    assert len(calls) == 1
    predicted = fitted.predict(features.build_features(sample))
    assert len(predicted) == len(sample)

    # default settings: random forest defaults with seed 42
    default = evaluate.refit_baseline(sample, lock)
    forest = default.named_steps["model"]
    assert forest.random_state == 42
    assert forest.n_estimators == 100
    assert forest.max_depth is None

    assert list(tmp_path.iterdir()) == []


def make_report_sample():
    sample = make_sample(12)
    sample["dropoff_at"] = sample["pickup_at"] + pd.Timedelta(minutes=15)
    return sample


def test_evaluation_does_not_need_baseline_settings(monkeypatch):
    """Seams stubbed: splits.load_test, splits.load_train, splits.draw_sample,
    train.read_lock, evaluate.evaluate_test and evaluate.refit_baseline.
    The champion loader is never called because evaluate_test is a stub.
    """
    from taxi_duration import evaluate, splits, train

    source = inspect.getsource(evaluate)
    assert "BASELINE_MODEL_URI" not in source
    assert "BASELINE_MODEL_SHA256" not in source

    monkeypatch.delenv("BASELINE_MODEL_URI", raising=False)
    monkeypatch.delenv("BASELINE_MODEL_SHA256", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///unused.db")
    monkeypatch.setenv("MODEL_URI", "models:/taxi-duration@champion")
    monkeypatch.setenv("MODEL_SHA256", "a" * 64)

    train_rows, sample = object(), make_report_sample()
    lock = {"sample_sha256": "x"}
    seen = {}

    def no_test_read():
        raise AssertionError("evaluate_test is stubbed and must not load rows")

    monkeypatch.setattr(splits, "load_test", no_test_read)
    monkeypatch.setattr(splits, "load_train", lambda: train_rows)

    def draw(df, *args, **kwargs):
        seen["drawn_from"] = df
        return sample

    monkeypatch.setattr(splits, "draw_sample", draw)
    monkeypatch.setattr(train, "read_lock", lambda *a, **k: lock)
    # run_evaluation will take train_mean from sample["duration_minutes"], get folds
    # from splits.make_folds(sample) and call evaluate.cv_reports(sample, folds, ...).
    # Both are stubbed here, so nothing is fitted.
    seen["cv_calls"] = []
    monkeypatch.setattr(
        splits, "make_folds", lambda frame: [{"train": [], "valid": []}]
    )
    monkeypatch.setattr(
        evaluate,
        "cv_reports",
        lambda *a, **k: seen["cv_calls"].append(a) or {},
        raising=False,
    )

    def refit(got_sample, got_lock, **kwargs):
        seen["refit"] = (got_sample, got_lock)
        return "fitted-baseline"

    monkeypatch.setattr(evaluate, "refit_baseline", refit, raising=False)

    def fake_evaluate_test(model_loader, baseline_loader, **kwargs):
        seen["baseline_model"] = baseline_loader()
        seen["kwargs"] = kwargs
        return {"test_reads": 1}

    monkeypatch.setattr(evaluate, "evaluate_test", fake_evaluate_test)

    assert evaluate.run_evaluation() == {"test_reads": 1}
    assert seen["drawn_from"] is train_rows
    assert seen["refit"][0] is sample
    assert seen["refit"][1] is lock
    assert seen["baseline_model"] == "fitted-baseline"
    assert seen["kwargs"]["load_test"] is splits.load_test
