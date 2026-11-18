"""Saving the model first checks that the disk has room for it."""

import shutil
from collections import namedtuple

import numpy as np
import pandas as pd

from taxi_duration import features

Usage = namedtuple("Usage", "total used free")


def small_fitted():
    rng = np.random.default_rng(3)
    n = 60
    distance = rng.uniform(0.5, 10, n)
    frame = pd.DataFrame(
        {
            "pickup_at": pd.Timestamp("2025-01-05")
            + pd.to_timedelta(np.arange(n) * 45, unit="m"),
            "pickup_zone_id": rng.choice([10, 20], n),
            "dropoff_zone_id": rng.choice([30, 40], n),
            "trip_distance": distance,
        }
    )
    pipe = features.make_pipeline().set_params(model__n_estimators=3)
    pipe.fit(features.build_features(frame), 3 + 2 * distance)
    return pipe


def test_save_refuses_when_disk_is_short(tmp_path, monkeypatch):
    import pytest

    from taxi_duration import artifacts

    fitted = small_fitted()
    folder = tmp_path / "model"

    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(100, 99, 1))
    with pytest.raises(RuntimeError, match="disk"):
        artifacts.save_model(fitted, folder)
    assert not folder.exists() or not any(folder.rglob("*.skops"))
    assert not folder.exists() or not any(folder.iterdir())

    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(10**12, 0, 10**12))
    size = artifacts.save_model(fitted, folder)
    assert size > 0
    assert (folder / artifacts.MODEL_FILE).is_file()
