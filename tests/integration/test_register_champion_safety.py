"""register_champion promotes only what it verified (review 2, finding 4).

Same local store as test_register_champion.py: sqlite and an artifact folder inside
tmp_path, no server, nothing written to ./mlruns.

Interface tune.register_champion must provide (signature unchanged):
- Before set_registered_model_alias it checks the registered files:
  verify_model.check_model_metadata on the logged or downloaded folder (called through
  the module, so a test can patch it) and the SHA-256 of the registered model.skops
  against the source file. Any failure propagates and the alias is not set or moved.
- Only an "already exists" MlflowException from create_registered_model is swallowed.
- Idempotent: when the alias champion already points to a version whose registered
  model.skops has the same SHA-256 as the source, it returns that version's dict (same
  keys as a new registration) and creates no run and no version.
- Different bytes create a new version and move the alias (the promotion case).
"""

import hashlib

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


def save(tmp_path, name, trees):
    frame = make_frame()
    pipe = features.make_pipeline().set_params(model__n_estimators=trees)
    pipe.fit(features.build_features(frame), frame["duration_minutes"])
    folder = tmp_path / name
    artifacts.save_model(pipe, folder)
    return folder


@pytest.fixture
def model_a(tmp_path):
    return save(tmp_path, "saved_a", 3)


@pytest.fixture
def model_b(tmp_path):
    return save(tmp_path, "saved_b", 4)


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
        return (
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


def raising(error):
    def check(*args, **kwargs):
        raise error

    return check


def test_failed_metadata_check_never_sets_the_alias(store, model_a, monkeypatch):
    from taxi_duration import tune

    calls = []

    def check(model_dir, trusted_types=frozenset()):
        calls.append(trusted_types)
        raise vm.ModelVerificationError("Loader 'x' is not allowed.")

    monkeypatch.setattr(vm, "check_model_metadata", check)

    with pytest.raises(vm.ModelVerificationError, match="not allowed"):
        tune.register_champion(model_a, 1_000, 1.0, tracking_uri=store)

    assert len(calls) == 1
    assert alias_version(store) is None


def test_failed_check_leaves_the_existing_champion(
    store, model_a, model_b, monkeypatch
):
    from taxi_duration import tune

    first = tune.register_champion(model_a, 1_000, 1.0, tracking_uri=store)
    monkeypatch.setattr(
        vm, "check_model_metadata", raising(vm.ModelVerificationError("bad metadata"))
    )

    with pytest.raises(vm.ModelVerificationError, match="bad metadata"):
        tune.register_champion(model_b, 1_000, 1.0, tracking_uri=store)

    assert str(alias_version(store)) == first["version"]


def test_corrupted_registered_file_never_gets_the_alias(
    store, model_a, tmp_path, monkeypatch
):
    """The bytes in the store differ from the source file: SHA-256 check fails."""
    import mlflow

    from taxi_duration import tune

    def corrupt_after(real):
        def wrapper(*args, **kwargs):
            out = real(*args, **kwargs)
            for stored in (tmp_path / "artifacts").rglob(artifacts.MODEL_FILE):
                with stored.open("ab") as handle:
                    handle.write(b"x")
            return out

        return wrapper

    monkeypatch.setattr(mlflow, "log_artifact", corrupt_after(mlflow.log_artifact))
    monkeypatch.setattr(
        MlflowClient, "log_artifact", corrupt_after(MlflowClient.log_artifact)
    )

    with pytest.raises(Exception):  # noqa: B017 (the exact type is the code's choice)
        tune.register_champion(model_a, 1_000, 1.0, tracking_uri=store)

    assert alias_version(store) is None


def test_other_mlflow_errors_are_not_swallowed(store, model_a, monkeypatch):
    from taxi_duration import tune

    # the model exists, so only a swallowed error lets the registration go on
    MlflowClient(store).create_registered_model(config.MODEL_NAME)
    monkeypatch.setattr(
        MlflowClient,
        "create_registered_model",
        raising(MlflowException("database is locked")),
    )

    with pytest.raises(MlflowException, match="database is locked"):
        tune.register_champion(model_a, 1_000, 1.0, tracking_uri=store)

    assert alias_version(store) is None


def test_registering_the_same_model_twice_makes_one_version(store, model_a):
    from taxi_duration import tune

    first = tune.register_champion(model_a, 1_000, 1.0, tracking_uri=store)
    runs = run_count(store)

    second = tune.register_champion(model_a, 1_000, 1.0, tracking_uri=store)

    assert second == first
    assert versions(store) == [first["version"]]
    assert run_count(store) == runs
    assert str(alias_version(store)) == first["version"]


def test_different_bytes_create_a_new_version_and_move_the_alias(
    store, model_a, model_b
):
    from taxi_duration import tune

    first = tune.register_champion(model_a, 1_000, 1.0, tracking_uri=store)
    assert (
        hashlib.sha256((model_a / artifacts.MODEL_FILE).read_bytes()).hexdigest()
        != hashlib.sha256((model_b / artifacts.MODEL_FILE).read_bytes()).hexdigest()
    )

    second = tune.register_champion(model_b, 1_000, 1.0, tracking_uri=store)

    assert second["version"] != first["version"]
    assert versions(store) == ["1", "2"]
    assert str(alias_version(store)) == second["version"]
    assert second["sha256"] != first["sha256"]
