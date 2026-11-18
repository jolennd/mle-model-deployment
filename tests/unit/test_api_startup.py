"""Startup loading, readiness and leak-free errors (ADR 8)."""

import pytest
from api_support import (
    MODEL_URI,
    PINNED_HASH,
    SECRET_PATH,
    TRACKING_URI,
    TRUSTED,
    VALID_REQUEST,
    VERSION,
    LoaderRecorder,
)
from fastapi.testclient import TestClient

from taxi_duration import verify_model


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("MODEL_URI", MODEL_URI)
    monkeypatch.setenv("MODEL_SHA256", PINNED_HASH)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", TRACKING_URI)
    monkeypatch.setenv("MODEL_TRUSTED_TYPES", TRUSTED)


def _install(monkeypatch, rec):
    monkeypatch.setattr(verify_model, "resolve_version", rec.resolve_version)
    monkeypatch.setattr(verify_model, "verify_then_load", rec.verify_then_load)


def test_model_loads_once_at_startup_with_pinned_settings(env, monkeypatch):
    from app.main import app

    rec = LoaderRecorder()
    _install(monkeypatch, rec)
    with TestClient(app) as client:
        assert len(rec.load_calls) == 1
        for _ in range(3):
            assert client.post("/predict", json=VALID_REQUEST).status_code == 200
    assert len(rec.load_calls) == 1
    assert rec.resolve_calls == [(MODEL_URI, TRACKING_URI)]
    call = rec.load_calls[0]
    assert call["model_uri"] == MODEL_URI
    assert call["expected_sha256"] == PINNED_HASH
    assert call["tracking_uri"] == TRACKING_URI
    assert call["trusted_types"] == frozenset({TRUSTED})
    assert call["version"] == VERSION


def test_not_ready_until_model_verified(env, monkeypatch):
    from app.main import app

    err = verify_model.ModelVerificationError(
        f"Hash mismatch: pinned {PINNED_HASH}, found {'0' * 64} in {SECRET_PATH}"
    )
    rec = LoaderRecorder(error=err)
    _install(monkeypatch, rec)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["ready"] is False
        response = client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 503
    assert len(rec.load_calls) == 1


@pytest.mark.parametrize("error", [OSError(SECRET_PATH), RuntimeError("boom")])
def test_any_load_failure_gives_503_not_500(env, monkeypatch, error):
    from app.main import app

    _install(monkeypatch, LoaderRecorder(error=error))
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/predict", json=VALID_REQUEST).status_code == 503


def test_errors_never_leak_hash_paths_or_traces(env, monkeypatch):
    from app.main import app

    err = verify_model.ModelVerificationError(
        f"Hash mismatch: pinned {PINNED_HASH}, found {'0' * 64} in {SECRET_PATH}"
    )
    _install(monkeypatch, LoaderRecorder(error=err))
    with TestClient(app) as client:
        texts = [
            client.post("/predict", json=VALID_REQUEST).text,
            client.get("/health").text,
        ]
    for text in texts:
        assert PINNED_HASH not in text
        assert "0" * 64 not in text
        assert SECRET_PATH not in text
        assert "/tmp" not in text
        assert "Traceback" not in text
        assert 'File "' not in text


def test_success_responses_never_leak_hash_or_paths(env, monkeypatch):
    from app.main import app

    _install(monkeypatch, LoaderRecorder())
    with TestClient(app) as client:
        texts = [
            client.post("/predict", json=VALID_REQUEST).text,
            client.get("/health").text,
        ]
    for text in texts:
        assert PINNED_HASH not in text
        assert "/tmp" not in text


def test_strict_tracking_uri_is_enforced_before_any_load(env, monkeypatch):
    from app.main import app

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example.com:5001")
    rec = LoaderRecorder()
    _install(monkeypatch, rec)
    with TestClient(app) as client:
        assert client.post("/predict", json=VALID_REQUEST).status_code == 503
    assert rec.resolve_calls == []
    assert rec.load_calls == []
