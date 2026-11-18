"""register_champion logs the exact skops file (batch 4).

A local sqlite store and an artifact folder inside tmp_path: no server, and nothing
is written to ./mlruns (ADR 8 point 11).

Interface tune.py must provide:
- register_champion(model_dir, size_bytes, predict_ms, tracking_uri=None) -> dict
  {"name": "taxi-duration", "alias": "champion", "version": str, "sha256": str}.
  Logs the model.skops file from model_dir byte for byte (no re-dump, no pickle),
  with the MLmodel metadata verify_model checks, registers taxi-duration, sets the
  alias champion, and returns the folder hash of the registered files
  (verify_model.model_sha256). Refuses (ValueError) before touching the store when
  passes_limits(size_bytes, predict_ms) is false.
"""

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from mlflow import MlflowClient

from taxi_duration import artifacts, config, features
from taxi_duration import verify_model as vm

pytestmark = pytest.mark.integration

TREE = "sklearn.tree._tree.Tree"
PICKLE_SUFFIXES = {".pkl", ".pickle", ".joblib"}


def make_frame(n=60):
    return pd.DataFrame(
        {
            "trip_id": [f"t{i:03d}" for i in range(n)],
            "pickup_at": pd.Timestamp("2025-01-05")
            + pd.to_timedelta([i * 30 for i in range(n)], unit="m"),
            "pickup_zone_id": [10, 20] * (n // 2),
            "dropoff_zone_id": [30, 40, 40, 30] * (n // 4),
            "trip_distance": [0.5 + (i % 12) for i in range(n)],
            "duration_minutes": [3.0 + 4 * (i % 12) for i in range(n)],
        }
    )


@pytest.fixture
def model_dir(tmp_path):
    frame = make_frame()
    pipe = features.make_pipeline().set_params(model__n_estimators=3)
    pipe.fit(features.build_features(frame), frame["duration_minutes"])
    folder = tmp_path / "saved"
    artifacts.save_model(pipe, folder)
    return folder


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_TRUSTED_TYPES", TREE)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    MlflowClient(uri).create_experiment(
        config.MODEL_NAME, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    return uri


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registers_the_exact_skops_file_and_pins_the_folder_hash(
    store, model_dir, tmp_path
):
    from taxi_duration import tune

    result = tune.register_champion(model_dir, 1_000, 1.0, tracking_uri=store)

    client = MlflowClient(store)
    version = client.get_model_version_by_alias(config.MODEL_NAME, "champion")
    local = vm.download(f"models:/{config.MODEL_NAME}@champion", store)

    registered = Path(local)
    files = [p for p in registered.rglob("*") if p.is_file()]
    skops_files = [p for p in files if p.name == artifacts.MODEL_FILE]
    assert len(skops_files) == 1
    assert file_sha(skops_files[0]) == file_sha(model_dir / artifacts.MODEL_FILE)
    assert not [
        p for p in files if p.suffix in PICKLE_SUFFIXES or "cloudpickle" in p.name
    ]

    assert result["alias"] == "champion"
    assert result["name"] == config.MODEL_NAME
    assert result["version"] == str(version.version)
    assert result["sha256"] == vm.model_sha256(registered)

    # What MODEL_SHA256 will pin must pass the verify step and load.
    vm.check_model_metadata(registered, {TREE})
    model, sha = vm.verify_then_load(
        f"models:/{config.MODEL_NAME}@champion",
        expected_sha256=result["sha256"],
        tracking_uri=store,
        trusted_types={TREE},
    )
    assert sha == result["sha256"]
    rows = features.build_features(make_frame().iloc[:2])
    assert len(model.predict(rows)) == 2


@pytest.mark.parametrize(
    ("size", "ms"),
    [
        (config.MODEL_SIZE_LIMIT_BYTES, 1.0),
        (1_000, config.MAX_PREDICT_MS),
    ],
)
def test_refuses_a_model_that_fails_a_limit(store, tmp_path, size, ms):
    from taxi_duration import tune

    with pytest.raises(ValueError):
        tune.register_champion(tmp_path / "no_such_model", size, ms, tracking_uri=store)

    assert MlflowClient(store).search_registered_models() == []
