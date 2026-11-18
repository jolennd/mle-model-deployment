"""Shared helpers for the API unit tests: a fake model and the loader recorder.

The tests replace the verify functions in taxi_duration.verify_model, so the app's
real startup path runs and no model store is needed.
"""

import numpy as np

PINNED_HASH = "deadbeef" * 8
MODEL_URI = "models:/taxi-duration@champion"
TRACKING_URI = "http://127.0.0.1:5001"
TRUSTED = "sklearn.tree._tree.Tree"
VERSION = "7"
SECRET_PATH = "/tmp/verified-model-abc123/model.skops"

VALID_REQUEST = {
    "pickup_zone_id": 161,
    "dropoff_zone_id": 237,
    "trip_distance": 2.4,
    "pickup_at": "2025-01-27T08:30:00",
}


class FakeModel:
    """Records every frame it is asked to score. Prediction is a fixed function of the features."""

    def __init__(self):
        self.seen = []

    def predict(self, frame):
        self.seen.append(frame.copy())
        return np.asarray(
            frame["trip_distance"] * 3.0
            + frame["pickup_hour"]
            + frame["pickup_dow"] * 0.5
            + frame["pickup_zone_id"] * 0.01
            + frame["dropoff_zone_id"] * 0.001,
            dtype=float,
        )


class LoaderRecorder:
    """Stands in for resolve_version and verify_then_load and records their calls."""

    def __init__(self, model=None, error=None):
        self.model = model if model is not None else FakeModel()
        self.error = error
        self.resolve_calls = []
        self.load_calls = []

    def resolve_version(self, model_uri, tracking_uri):
        self.resolve_calls.append((model_uri, tracking_uri))
        return VERSION

    def verify_then_load(
        self,
        model_uri,
        *,
        expected_sha256,
        tracking_uri,
        trusted_types=frozenset(),
        allowed_hosts=None,
        version=None,
    ):
        self.load_calls.append(
            {
                "model_uri": model_uri,
                "expected_sha256": expected_sha256,
                "tracking_uri": tracking_uri,
                "trusted_types": frozenset(trusted_types),
                "version": version,
            }
        )
        if self.error is not None:
            raise self.error
        return self.model, expected_sha256
