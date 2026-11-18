"""The baseline is measured in memory and never saved again (ADR 8 point 15).

Interface to implement in tune.py:
- BASELINE_SIZE_BYTES = 2_694_430_930
- baseline_record(lock, predict_ms) -> {"name": "baseline", "fold_rmse":
  lock["fold_scores"], "size_bytes": BASELINE_SIZE_BYTES, "predict_ms": predict_ms}
- measure_baseline_ms(sample, folds) -> float: fits the fold 3 baseline pipeline in
  memory, sets n_jobs=1, returns measure_predict_ms on one validation row.
- run_tuning builds the baseline with these two and no longer measures it with
  _fold3_measure.
"""

import numpy as np
import pandas as pd


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


def test_baseline_measured_in_memory(monkeypatch):
    from taxi_duration import artifacts, tune

    def boom(*args, **kwargs):
        raise AssertionError("the baseline must not be saved or loaded")

    monkeypatch.setattr(artifacts, "save_model", boom)
    monkeypatch.setattr(artifacts, "dump", boom, raising=False)
    monkeypatch.setattr(artifacts, "load_model", boom)

    assert tune.BASELINE_SIZE_BYTES == 2_694_430_930

    lock = {"fold_scores": [4.0, 3.9, 3.8]}
    record = tune.baseline_record(lock, 12.5)
    assert record == {
        "name": "baseline",
        "fold_rmse": [4.0, 3.9, 3.8],
        "size_bytes": 2_694_430_930,
        "predict_ms": 12.5,
    }

    df = make_frame()
    ids = df["trip_id"].tolist()
    folds = [
        {"train": ids[:120], "valid": ids[120:170]},
        {"train": ids[:170], "valid": ids[170:220]},
        {"train": ids[:220], "valid": ids[220:270]},
    ]
    ms = tune.measure_baseline_ms(df, folds)
    assert isinstance(ms, float)
    assert 0 < ms < 5000


class FakeModel:
    """Stands in for a reloaded pipeline so timing code can run without a real file."""

    def __init__(self):
        self.named_steps = {"model": self}

    def set_params(self, **kwargs):
        return self

    def predict(self, rows):
        return np.zeros(len(rows))


def test_run_tuning_never_saves_the_baseline(monkeypatch, tmp_path):
    import mlflow

    from taxi_duration import artifacts, config, splits, train, tune

    saves = []

    def record_save(*args, **kwargs):
        saves.append(args)
        return 1

    monkeypatch.setattr(artifacts, "save_model", record_save)
    monkeypatch.setattr(artifacts, "dump", record_save, raising=False)
    monkeypatch.setattr(artifacts, "load_model", lambda *a, **k: FakeModel())

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    monkeypatch.setenv("MODEL_TRUSTED_TYPES", "numpy.ndarray")
    monkeypatch.setattr(mlflow, "MlflowClient", lambda *a, **k: object())
    monkeypatch.setattr(train, "_experiment_id", lambda client, uri: "exp")
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(tune, "MODEL_DIR", str(tmp_path / "model"))

    df = make_frame()
    ids = df["trip_id"].tolist()
    folds = [
        {"train": ids[:120], "valid": ids[120:170]},
        {"train": ids[:170], "valid": ids[170:220]},
        {"train": ids[:220], "valid": ids[220:270]},
    ]
    monkeypatch.setattr(splits, "load_train", lambda: df)
    monkeypatch.setattr(splits, "draw_sample", lambda frame: frame)
    monkeypatch.setattr(splits, "make_folds", lambda frame: folds)
    monkeypatch.setattr(
        train,
        "read_lock",
        lambda: {"fold_scores": [4.0, 3.9, 3.8], "sample_sha256": "x"},
    )
    monkeypatch.setattr(tune, "run_study", lambda study, objective, **kw: None)

    timed = []

    def fake_measure(sample, fold_list):
        timed.append((sample, fold_list))
        return 1.0

    monkeypatch.setattr(tune, "measure_baseline_ms", fake_measure, raising=False)

    # No trials run, so no candidate passes the size limit (ADR 8 point 14).
    # The save assertion comes first so a failure names the real cause.
    error = None
    try:
        tune.run_tuning()
    except RuntimeError as exc:
        error = exc

    assert saves == [], f"the baseline was saved {len(saves)} time(s)"
    assert error is not None and "No candidate passes" in str(error)
    assert len(timed) == 1
    assert timed[0][0] is df
    assert timed[0][1] is folds
