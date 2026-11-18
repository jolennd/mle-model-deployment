"""The API against the real registered champion.

Needs the local model store on 127.0.0.1:5001 and MODEL_SHA256 set in the
environment (.env). Skips with a stated reason when either is missing.
"""

import os
import socket

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from taxi_duration import verify_model
from taxi_duration.features import build_features

pytestmark = pytest.mark.model

REQUEST = {
    "pickup_zone_id": 161,
    "dropoff_zone_id": 237,
    "trip_distance": 2.4,
    "pickup_at": "2025-01-27T08:30:00",
}


def _store_up():
    try:
        with socket.create_connection(("127.0.0.1", 5001), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def client():
    for name in ("MODEL_SHA256", "MODEL_URI", "MLFLOW_TRACKING_URI"):
        if not os.environ.get(name):
            pytest.skip(f"{name} is not set (put it in .env and run through make)")
    if not _store_up():
        pytest.skip("model store is not running on 127.0.0.1:5001 (make mlflow-server)")
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_champion_is_ready_and_answers(client):
    assert client.get("/health").json()["ready"] is True
    response = client.post("/predict", json=REQUEST)
    assert response.status_code == 200
    body = response.json()
    minutes = body["prediction_minutes"]
    assert 1 <= minutes <= 60
    assert body["model_version"] == str(
        verify_model.resolve_version(
            os.environ["MODEL_URI"], os.environ["MLFLOW_TRACKING_URI"]
        )
    )
    assert os.environ["MODEL_SHA256"] not in response.text


def test_champion_prediction_matches_training_code(client):
    model, _ = verify_model.verify_then_load(
        os.environ["MODEL_URI"],
        expected_sha256=os.environ["MODEL_SHA256"],
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        trusted_types=frozenset(
            t.strip()
            for t in os.environ.get("MODEL_TRUSTED_TYPES", "").split(",")
            if t.strip()
        ),
    )
    expected = float(model.predict(build_features(pd.DataFrame([REQUEST])))[0])
    got = client.post("/predict", json=REQUEST).json()["prediction_minutes"]
    assert got == pytest.approx(expected, abs=1e-6)
