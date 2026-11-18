"""The sample request in the README passes the API's request validation."""

import json
from pathlib import Path

from api_support import (
    MODEL_URI,
    PINNED_HASH,
    TRACKING_URI,
    TRUSTED,
    LoaderRecorder,
)
from fastapi.testclient import TestClient

from taxi_duration import verify_model

ROOT = Path(__file__).resolve().parents[2]


def test_sample_request_matches_schema(monkeypatch):
    path = ROOT / "sample-request.json"
    assert path.exists(), "sample-request.json is missing"
    payload = json.loads(path.read_text())
    rec = LoaderRecorder()
    monkeypatch.setenv("MODEL_URI", MODEL_URI)
    monkeypatch.setenv("MODEL_SHA256", PINNED_HASH)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", TRACKING_URI)
    monkeypatch.setenv("MODEL_TRUSTED_TYPES", TRUSTED)
    monkeypatch.setattr(verify_model, "resolve_version", rec.resolve_version)
    monkeypatch.setattr(verify_model, "verify_then_load", rec.verify_then_load)
    from app.main import app

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code != 422, response.text
    assert response.status_code == 200
