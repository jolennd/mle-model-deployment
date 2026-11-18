"""The model must beat the mean reference on every fold."""

import numpy as np
import pandas as pd

from taxi_duration import features


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


def test_baseline_beats_naive_reference():
    from taxi_duration import train

    df = make_frame()
    ids = df["trip_id"].tolist()
    folds = [
        {"train": ids[:120], "valid": ids[120:170]},
        {"train": ids[:170], "valid": ids[170:220]},
        {"train": ids[:220], "valid": ids[220:270]},
    ]
    calls = []

    def factory():
        calls.append(1)
        return features.make_pipeline().set_params(model__n_estimators=10)

    cv = train.cross_validate(df, folds, pipeline_factory=factory)

    assert len(calls) == len(folds)
    assert len(cv["model"]) == len(cv["mean_reference"]) == len(folds)
    for model_rmse, ref_rmse in zip(cv["model"], cv["mean_reference"], strict=True):
        assert isinstance(model_rmse, float)
        assert isinstance(ref_rmse, float)
        assert model_rmse < ref_rmse
