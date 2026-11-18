"""A pickup time with a time zone is a bad request (422), never a server error."""

import pytest
from api_support import (
    MODEL_URI,
    PINNED_HASH,
    TRACKING_URI,
    TRUSTED,
    VALID_REQUEST,
    LoaderRecorder,
)
from fastapi.testclient import TestClient

from taxi_duration import verify_model


@pytest.fixture
def client(monkeypatch):
    rec = LoaderRecorder()
    monkeypatch.setenv("MODEL_URI", MODEL_URI)
    monkeypatch.setenv("MODEL_SHA256", PINNED_HASH)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", TRACKING_URI)
    monkeypatch.setenv("MODEL_TRUSTED_TYPES", TRUSTED)
    monkeypatch.setattr(verify_model, "resolve_version", rec.resolve_version)
    monkeypatch.setattr(verify_model, "verify_then_load", rec.verify_then_load)
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize(
    "pickup_at",
    ["2025-01-27T08:30:00+01:00", "2025-01-27T08:30:00Z", "2025-01-27T08:30:00-05:00"],
)
def test_timezone_aware_pickup_returns_422(client, pickup_at):
    response = client.post("/predict", json={**VALID_REQUEST, "pickup_at": pickup_at})
    assert response.status_code == 422
    assert "Traceback" not in response.text


def test_naive_pickup_still_accepted(client):
    assert client.post("/predict", json=VALID_REQUEST).status_code == 200
