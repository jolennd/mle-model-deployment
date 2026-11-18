"""Model artifact: safe format, single-threaded forest, trusted types only."""

import os

import numpy as np
import pandas as pd
import pytest

from taxi_duration import features

pytestmark = pytest.mark.integration


def make_frame(n=120):
    rng = np.random.default_rng(2)
    distance = rng.uniform(0.5, 10, n)
    return pd.DataFrame(
        {
            "pickup_at": pd.Timestamp("2025-01-05")
            + pd.to_timedelta(np.arange(n) * 45, unit="m"),
            "pickup_zone_id": rng.choice([10, 20, 30], n),
            "dropoff_zone_id": rng.choice([40, 50], n),
            "trip_distance": distance,
            "duration_minutes": 3 + 2 * distance + rng.normal(0, 0.2, n),
        }
    )


def test_artifact_round_trip(tmp_path, monkeypatch):
    from taxi_duration import artifacts

    monkeypatch.setenv("MODEL_TRUSTED_TYPES", "sklearn.tree._tree.Tree")
    trusted = os.environ["MODEL_TRUSTED_TYPES"].split(",")

    frame = make_frame()
    pipe = features.make_pipeline().set_params(model__n_estimators=5, model__n_jobs=-1)
    pipe.fit(features.build_features(frame), frame["duration_minutes"])
    expected = pipe.predict(features.build_features(frame))

    folder = tmp_path / "model"
    folder.mkdir()
    size = artifacts.save_model(pipe, folder)

    written = [p for p in folder.rglob("*") if p.is_file()]
    assert size > 0
    assert size == sum(p.stat().st_size for p in written)

    loaded = artifacts.load_model(folder, trusted)
    actual = loaded.predict(features.build_features(frame))
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    assert loaded.named_steps["model"].n_jobs == 1

    with pytest.raises(Exception):  # noqa: B017 - skops raises its own error type
        artifacts.load_model(folder, [])
