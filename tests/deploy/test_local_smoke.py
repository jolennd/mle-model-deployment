"""Live request against a running API. Skips without API_URL.

Run it after starting the service, for example:
API_URL=http://127.0.0.1:8000 uv run pytest -m deploy
"""

import json
import os
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.deploy

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def api_url():
    url = os.environ.get("API_URL")
    if not url:
        pytest.skip("API_URL is not set (start the service and set it to its address)")
    return url.rstrip("/")


def test_live_predict_request(api_url):
    payload = json.loads((ROOT / "sample-request.json").read_text())
    response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction_minutes", "model_version"}
    assert 1 <= body["prediction_minutes"] <= 60
