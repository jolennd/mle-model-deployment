"""Prediction API contract with a fake model (ADR 2, ADR 8).

Input names follow the training columns: pickup_zone_id, dropoff_zone_id,
trip_distance, pickup_at. The app is app.main:app and reads its settings from
the environment when it starts.
"""

import pandas as pd
import pytest
from api_support import (
    MODEL_URI,
    PINNED_HASH,
    TRACKING_URI,
    TRUSTED,
    VALID_REQUEST,
    VERSION,
    FakeModel,
    LoaderRecorder,
)
from fastapi.testclient import TestClient

from taxi_duration import config, verify_model
from taxi_duration.features import build_features


@pytest.fixture
def loader(monkeypatch):
    rec = LoaderRecorder()
    monkeypatch.setenv("MODEL_URI", MODEL_URI)
    monkeypatch.setenv("MODEL_SHA256", PINNED_HASH)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", TRACKING_URI)
    monkeypatch.setenv("MODEL_TRUSTED_TYPES", TRUSTED)
    monkeypatch.setattr(verify_model, "resolve_version", rec.resolve_version)
    monkeypatch.setattr(verify_model, "verify_then_load", rec.verify_then_load)
    return rec


@pytest.fixture
def client(loader):
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_api_max_distance_comes_from_adr_2():
    assert config.API_MAX_DISTANCE_MILES == 80


def test_valid_request_returns_prediction_and_version(client):
    response = client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction_minutes", "model_version"}
    assert isinstance(body["prediction_minutes"], float)
    assert body["model_version"] == VERSION


@pytest.mark.parametrize(
    "changes",
    [
        {"pickup_zone_id": 0},
        {"pickup_zone_id": 264},
        {"pickup_zone_id": 266},
        {"dropoff_zone_id": 0},
        {"dropoff_zone_id": 264},
        {"dropoff_zone_id": 266},
        {"trip_distance": 0.09},
        {"trip_distance": 80.01},
        {"trip_distance": -1},
        {"pickup_at": "not a date"},
        {"pickup_at": None},
        {"trip_distance": "far"},
    ],
)
def test_invalid_input_returns_422(client, changes):
    response = client.post("/predict", json={**VALID_REQUEST, **changes})
    assert response.status_code == 422
    assert "Traceback" not in response.text


@pytest.mark.parametrize("missing", sorted(VALID_REQUEST))
def test_missing_field_returns_422(client, missing):
    payload = {k: v for k, v in VALID_REQUEST.items() if k != missing}
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "changes",
    [
        {"pickup_zone_id": 1},
        {"pickup_zone_id": 263},
        {"dropoff_zone_id": 265},
        {"trip_distance": 0.1},
        {"trip_distance": 80},
    ],
)
def test_bounds_are_inclusive(client, changes):
    response = client.post("/predict", json={**VALID_REQUEST, **changes})
    assert response.status_code == 200


def test_health_answers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_features_match_training(client, loader):
    """The model sees exactly what build_features gives for the same row."""
    assert client.post("/predict", json=VALID_REQUEST).status_code == 200
    row = pd.DataFrame(
        [
            {
                "trip_distance": VALID_REQUEST["trip_distance"],
                "pickup_at": VALID_REQUEST["pickup_at"],
                "pickup_zone_id": VALID_REQUEST["pickup_zone_id"],
                "dropoff_zone_id": VALID_REQUEST["dropoff_zone_id"],
            }
        ]
    )
    expected = build_features(row)
    seen = loader.model.seen
    assert len(seen) == 1
    assert list(seen[0].columns) == list(config.FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(seen[0].reset_index(drop=True), expected)


def test_prediction_matches_training_within_1e6(client, loader):
    response = client.post("/predict", json=VALID_REQUEST)
    row = pd.DataFrame([VALID_REQUEST])
    expected = FakeModel().predict(build_features(row))[0]
    assert response.json()["prediction_minutes"] == pytest.approx(expected, abs=1e-6)
