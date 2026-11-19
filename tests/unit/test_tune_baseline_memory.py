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
import pytest
from tune_run_support import setup_run


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
    """Seams: tune._fit, tune.score_params, tune.measure_predict_ms, tune.run_study,
    tune.prepare_model_dir, artifacts.save_model and load_model (via setup_run).

    Intent "the baseline is never saved": zero trials, so the baseline is the only
    candidate and fails the size limit. The fallback then saves capped models. Every
    saved model must carry capped params (a depth), never the baseline's {} settings,
    and the baseline timing must come from measure_baseline_ms on the same objects.
    """
    from taxi_duration import splits, tune

    saved_params = []

    def size_for(p):
        saved_params.append(dict(p))
        return 1_000_000

    seen = setup_run(monkeypatch, tmp_path, trials=(), size_for=size_for)
    df = splits.load_train()
    folds = splits.make_folds(df)
    timed = []

    def fake_measure(sample, fold_list):
        timed.append((sample, fold_list))
        return 1.0

    monkeypatch.setattr(tune, "measure_baseline_ms", fake_measure)

    with pytest.raises(tune.NeedsApproval):
        tune.run_tuning()

    assert saved_params, "the fallback saved no capped model"
    assert all(p.get("max_depth") is not None for p in saved_params), (
        f"a model without a depth cap was saved: {saved_params}"
    )
    assert {} not in seen.fits
    assert len(timed) == 1
    assert timed[0][0] is df
    assert timed[0][1] is folds
