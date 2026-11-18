"""Behavioral regression tests for the saved model: minimum and directional.

Cases live in cases.toml next to this file, approved by the project owner. These tests
load the registered model by alias, never train and never touch the test week.
Settings come from the environment: MLFLOW_TRACKING_URI, MODEL_URI (models:/name@alias),
MODEL_SHA256 and MODEL_TRUSTED_TYPES (comma separated). If one is unset, or the
server or alias is missing, the tests skip with the reason.
"""

import os
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from taxi_duration import features

pytestmark = pytest.mark.behavioral

CASES = tomllib.loads((Path(__file__).parent / "cases.toml").read_text())["case"]
SETTINGS = ("MLFLOW_TRACKING_URI", "MODEL_URI", "MODEL_SHA256")


def cases(kind: str) -> list:
    out = []
    for case in CASES:
        if case["type"] != kind:
            continue
        marks = (
            [pytest.mark.xfail(strict=True, reason=case["xfail"])]
            if "xfail" in case
            else []
        )
        out.append(pytest.param(case, id=case["id"], marks=marks))
    return out


@pytest.fixture(scope="module")
def fitted():
    from taxi_duration import verify_model

    missing = [name for name in SETTINGS if not os.environ.get(name)]
    if missing:
        pytest.skip(f"behavioral model not configured: {', '.join(missing)} unset")
    trusted = [
        t for t in os.environ.get("MODEL_TRUSTED_TYPES", "").split(",") if t.strip()
    ]
    try:
        model, _ = verify_model.verify_then_load(
            os.environ["MODEL_URI"],
            expected_sha256=os.environ["MODEL_SHA256"],
            tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            trusted_types=frozenset(trusted),
        )
    except (MlflowException, OSError) as exc:
        pytest.skip(f"model server or alias not reachable: {exc}")
    return model


def predict(fitted, inputs: dict) -> float:
    """Predicted minutes for one trip given as the four API fields."""
    frame = pd.DataFrame([inputs])
    frame["pickup_at"] = pd.to_datetime(frame["pickup_at"])
    return float(fitted.predict(features.build_features(frame))[0])


@pytest.mark.parametrize("case", cases("mft"))
def test_minimum_functionality(fitted, case):
    got = predict(fitted, case["input"])
    assert got >= case["expected_min"], (
        f"{case['id']}: {case['input']!r} gave {got:.2f} minutes, "
        f"expected at least {case['expected_min']}"
    )


@pytest.mark.parametrize("case", cases("dir"))
def test_direction(fitted, case):
    before = predict(fitted, case["input"])
    after = predict(fitted, case["perturbed"])
    sign = 1 if case["direction"] == "up" else -1
    assert sign * (after - before) >= case["min_step"], (
        f"{case['id']}: {case['input']!r} -> {before:.2f}; "
        f"{case['perturbed']!r} -> {after:.2f}; "
        f"expected {case['direction']} by at least {case['min_step']}"
    )
