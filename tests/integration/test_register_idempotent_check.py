"""Idempotent re-register re-runs the metadata check (ADR 8 point 21), review 3.

Same local store as test_register_champion_safety.py: sqlite and an artifact folder
inside tmp_path, no server, nothing written to ./mlruns.

Interface tune.register_champion must provide:
- When the alias champion already points to a version with the same model.skops bytes,
  it calls verify_model.check_model_metadata (through the module attribute, so a test can
  patch it) on the downloaded folder with the project allow list as a set, and only
  then returns the existing version's dict.
- If that check raises, the exception propagates, no run and no version is created and
  the alias stays where it was.
"""

import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from taxi_duration import artifacts, config, features
from taxi_duration import verify_model as vm

pytestmark = pytest.mark.integration

TREE = "sklearn.tree._tree.Tree"


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


def alias_version(store):
    try:
        return str(
            MlflowClient(store)
            .get_model_version_by_alias(config.MODEL_NAME, "champion")
            .version
        )
    except MlflowException:
        return None


def versions(store):
    found = MlflowClient(store).search_model_versions(f"name='{config.MODEL_NAME}'")
    return sorted(str(v.version) for v in found)


def run_count(store):
    client = MlflowClient(store)
    experiment = client.get_experiment_by_name(config.MODEL_NAME)
    return len(client.search_runs([experiment.experiment_id]))


def test_idempotent_register_runs_the_metadata_check(store, model_dir, monkeypatch):
    from taxi_duration import tune

    first = tune.register_champion(model_dir, 1_000, 1.0, tracking_uri=store)
    real = vm.check_model_metadata
    calls = []

    def check(folder, trusted_types=frozenset()):
        calls.append(set(trusted_types))
        return real(folder, trusted_types)

    monkeypatch.setattr(vm, "check_model_metadata", check)

    second = tune.register_champion(model_dir, 1_000, 1.0, tracking_uri=store)

    assert second == first
    assert calls == [{TREE}]


def test_idempotent_register_with_a_failing_check_changes_nothing(
    store, model_dir, monkeypatch
):
    from taxi_duration import tune

    first = tune.register_champion(model_dir, 1_000, 1.0, tracking_uri=store)
    runs = run_count(store)

    def check(folder, trusted_types=frozenset()):
        raise vm.ModelVerificationError("bad metadata on the existing version")

    monkeypatch.setattr(vm, "check_model_metadata", check)

    with pytest.raises(vm.ModelVerificationError, match="bad metadata"):
        tune.register_champion(model_dir, 1_000, 1.0, tracking_uri=store)

    assert versions(store) == [first["version"]]
    assert run_count(store) == runs
    assert alias_version(store) == first["version"]
