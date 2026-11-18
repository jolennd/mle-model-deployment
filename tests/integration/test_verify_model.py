"""Tests for verify_model. Copy to tests/test_verify_model.py.

They build small models on the spot in a temporary folder: no server, no network.
"""

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from taxi_duration import verify_model as vm

TREE = "sklearn.tree._tree.Tree"


def _fit(estimator):
    X = np.arange(40, dtype=float).reshape(20, 2)
    return estimator.fit(X, X.sum(axis=1))


@pytest.fixture
def safe_dir(tmp_path):
    path = tmp_path / "model"
    mlflow.sklearn.save_model(
        _fit(LinearRegression()), path, serialization_format="skops"
    )
    return path


def test_safe_model_passes_and_hash_is_stable(safe_dir):
    sha = vm.model_sha256(safe_dir)
    assert vm.verify_local_dir(safe_dir, sha) == sha
    assert vm.model_sha256(safe_dir) == sha


def test_hash_mismatch_is_refused(safe_dir):
    with pytest.raises(vm.ModelVerificationError, match="mismatch"):
        vm.verify_local_dir(safe_dir, "0" * 64)


def test_missing_pinned_hash_is_refused(safe_dir):
    with pytest.raises(vm.ModelVerificationError, match="No pinned hash"):
        vm.verify_local_dir(safe_dir, "")


def test_renaming_a_file_changes_the_hash(safe_dir):
    before = vm.model_sha256(safe_dir)
    (safe_dir / "requirements.txt").rename(safe_dir / "requirements.old")
    assert vm.model_sha256(safe_dir) != before


def test_pickle_format_is_refused(tmp_path):
    path = tmp_path / "pickled"
    mlflow.sklearn.save_model(
        _fit(LinearRegression()), path, serialization_format="cloudpickle"
    )
    with pytest.raises(vm.ModelVerificationError, match="skops"):
        vm.check_model_metadata(path)


def test_code_folder_is_refused(safe_dir):
    (safe_dir / "code").mkdir()
    with pytest.raises(vm.ModelVerificationError, match="code"):
        vm.check_model_metadata(safe_dir)


def test_types_the_model_trusts_must_be_in_the_project_allow_list(tmp_path):
    path = tmp_path / "forest"
    mlflow.sklearn.save_model(
        _fit(RandomForestRegressor(n_estimators=2, random_state=0)),
        path,
        serialization_format="skops",
        skops_trusted_types=[TREE],
    )
    with pytest.raises(vm.ModelVerificationError, match="allow list"):
        vm.check_model_metadata(path)
    vm.check_model_metadata(path, trusted_types={TREE})


@pytest.mark.parametrize(
    "uri", ["models:/taxi/3", "models:/taxi/latest", "runs:/abc/model", "/tmp/m"]
)
def test_only_alias_uris_are_accepted(uri):
    with pytest.raises(vm.ModelVerificationError, match="alias"):
        vm.check_alias_uri(uri)


@pytest.mark.parametrize(
    "uri", ["http://10.0.0.5:5000", "https://example.com", "databricks"]
)
def test_remote_tracking_servers_are_refused(uri):
    with pytest.raises(vm.ModelVerificationError, match="not a local store"):
        vm.check_tracking_uri(uri)


@pytest.mark.parametrize(
    "uri", ["http://127.0.0.1:5000", "http://localhost:5001", "sqlite:///m.db"]
)
def test_local_tracking_servers_are_accepted(uri):
    vm.check_tracking_uri(uri)


def test_verify_then_load_by_alias_end_to_end(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    # Artifacts stay inside tmp_path, never ./mlruns in the repo (ADR 8 point 11).
    mlflow.create_experiment(
        "verify-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment("verify-test")
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            _fit(LinearRegression()),
            name="model",
            serialization_format="skops",
            registered_model_name="taxi",
        )
    MlflowClient(tracking_uri).set_registered_model_alias(
        "taxi", "champion", info.registered_model_version
    )
    sha = vm.model_sha256(vm.download("models:/taxi@champion", tracking_uri))

    model, got = vm.verify_then_load(
        "models:/taxi@champion", expected_sha256=sha, tracking_uri=tracking_uri
    )

    assert got == sha
    assert model.predict(np.array([[1.0, 2.0]])).shape == (1,)
