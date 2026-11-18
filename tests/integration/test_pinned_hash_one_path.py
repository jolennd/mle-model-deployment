"""One pinned hash for every load path (ADR 8 point 20), review 3.

A real small registered model in a sqlite store and an artifact folder inside
tmp_path: no server, nothing written to ./mlruns. mlflow writes a client-side file
registered_model_meta into every folder downloaded from models:/, and its text holds
the version in a different form for an alias download and a version download. So the
hash of the two folders differs although model.skops is identical.

Seam: a spy replaces mlflow.MlflowClient (and verify_model.MlflowClient if the module
has it) with a subclass that always talks to the tmp_path store and records
get_model_version_by_alias calls, and wraps mlflow.artifacts.download_artifacts to
record the artifact_uri of every download. This lets verify_model.main run with the
strict environment address without a server.

Interface the code must provide:
- Every load path (tune.register_champion, verify_model.main, verify_model.
  verify_then_load, the evaluation load) does ONE alias lookup, then downloads with the
  version pinned (artifact_uri models:/<name>/<version>, never models:/<name>@<alias>),
  and hashes that download with verify_model.model_sha256 (unchanged).
- verify_then_load(alias_uri, ...) without version resolves the version itself, once,
  and pins it. With version given it does no lookup.
- verify_model.main checks the metadata against the MODEL_TRUSTED_TYPES list of the
  environment (today it passes an empty set and refuses the real champion).
- register_champion returns and prints the hash of the pinned download of the version
  that holds the alias.
"""

import re
from types import SimpleNamespace

import mlflow
import pytest
from mlflow import MlflowClient

from taxi_duration import artifacts, config, features
from taxi_duration import verify_model as vm

pytestmark = pytest.mark.integration

TREE = "sklearn.tree._tree.Tree"
ALIAS_URI = f"models:/{config.MODEL_NAME}@champion"
GOOD_ENV_URI = "http://127.0.0.1:5001"


def make_frame(n=60):
    import pandas as pd

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


@pytest.fixture
def spy(store, monkeypatch):
    calls = SimpleNamespace(alias=[], downloads=[])
    real_client = MlflowClient
    real_download = mlflow.artifacts.download_artifacts

    class SpyClient(real_client):
        def __init__(self, tracking_uri=None, *args, **kwargs):
            super().__init__(store, *args, **kwargs)

        def get_model_version_by_alias(self, name, alias):
            calls.alias.append((name, alias))
            return super().get_model_version_by_alias(name, alias)

    def download(*args, **kwargs):
        uri = kwargs.get("artifact_uri") or args[0]
        calls.downloads.append(uri)
        kwargs["tracking_uri"] = store
        return real_download(*args, **kwargs)

    monkeypatch.setattr(mlflow, "MlflowClient", SpyClient)
    monkeypatch.setattr(vm, "MlflowClient", SpyClient, raising=False)
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download)
    return calls


def reset(spy):
    spy.alias.clear()
    spy.downloads.clear()


def printed_hash(text):
    found = re.findall(r"^MODEL_SHA256=([0-9a-f]{64})$", text, flags=re.MULTILINE)
    assert len(found) == 1, text
    return found[0]


@pytest.fixture
def registered(store, spy, model_dir, capsys, monkeypatch):
    """Register once; return the result and the hash register_champion printed."""
    from taxi_duration import tune

    result = tune.register_champion(model_dir, 1_000, 1.0, tracking_uri=store)
    out = capsys.readouterr().out
    reset(spy)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", GOOD_ENV_URI)
    return SimpleNamespace(result=result, printed=printed_hash(out))


def run_main(capsys):
    assert vm.main([ALIAS_URI]) == 0
    return printed_hash(capsys.readouterr().out)


def test_register_champion_prints_the_hash_it_returns(registered):
    assert registered.printed == registered.result["sha256"]
    assert registered.result["version"] == "1"


def test_main_prints_the_same_hash_as_register(registered, spy, capsys):
    assert run_main(capsys) == registered.result["sha256"]


def test_verify_then_load_by_alias_accepts_the_register_hash(registered, spy, store):
    model, sha = vm.verify_then_load(
        ALIAS_URI,
        expected_sha256=registered.result["sha256"],
        tracking_uri=store,
        trusted_types={TREE},
    )

    assert sha == registered.result["sha256"]
    assert len(model.predict(features.build_features(make_frame().iloc[:2]))) == 2


def test_evaluation_load_path_accepts_the_register_hash(registered, spy, store):
    """What evaluate.run_evaluation does: one lookup, then a pinned verify_then_load."""
    version = str(
        MlflowClient(store)
        .get_model_version_by_alias(config.MODEL_NAME, "champion")
        .version
    )

    _, sha = vm.verify_then_load(
        ALIAS_URI,
        expected_sha256=registered.result["sha256"],
        tracking_uri=store,
        trusted_types={TREE},
        version=version,
    )

    assert sha == registered.result["sha256"]


def assert_one_lookup_then_pinned_download(spy, version="1"):
    assert spy.alias == [(config.MODEL_NAME, "champion")]
    assert spy.downloads == [f"models:/{config.MODEL_NAME}/{version}"]


def test_main_does_one_alias_lookup_then_a_pinned_download(registered, spy, capsys):
    run_main(capsys)

    assert_one_lookup_then_pinned_download(spy)


def test_verify_then_load_by_alias_resolves_the_version_once_and_pins_it(
    registered, spy, store
):
    vm.verify_then_load(
        ALIAS_URI,
        expected_sha256=registered.result["sha256"],
        tracking_uri=store,
        trusted_types={TREE},
    )

    assert_one_lookup_then_pinned_download(spy)


def test_verify_then_load_with_a_version_does_no_lookup(registered, spy, store):
    vm.verify_then_load(
        ALIAS_URI,
        expected_sha256=registered.result["sha256"],
        tracking_uri=store,
        trusted_types={TREE},
        version="1",
    )

    assert spy.alias == []
    assert spy.downloads == [f"models:/{config.MODEL_NAME}/1"]


def test_register_champion_never_downloads_by_alias(store, spy, model_dir):
    from taxi_duration import tune

    result = tune.register_champion(model_dir, 1_000, 1.0, tracking_uri=store)

    assert spy.downloads, "no download seen"
    assert not [uri for uri in spy.downloads if "@" in uri], spy.downloads
    assert spy.downloads[-1] == f"models:/{config.MODEL_NAME}/{result['version']}"


def test_idempotent_register_does_one_lookup_and_a_pinned_download(
    registered, spy, store, model_dir
):
    from taxi_duration import tune

    again = tune.register_champion(model_dir, 1_000, 1.0, tracking_uri=store)

    assert again == registered.result
    assert_one_lookup_then_pinned_download(spy)


def register_second_version_of_the_same_bytes(store, tmp_path):
    """A new run and a new version with the same MLmodel and model.skops, no alias move."""
    local = vm.download(ALIAS_URI, store, version=1)
    client = MlflowClient(store)
    experiment = client.get_experiment_by_name(config.MODEL_NAME)
    run = client.create_run(experiment.experiment_id)
    for name in ("MLmodel", artifacts.MODEL_FILE):
        client.log_artifact(run.info.run_id, f"{local}/{name}", artifact_path="model")
    client.set_terminated(run.info.run_id)
    version = client.create_model_version(
        config.MODEL_NAME, f"{run.info.artifact_uri}/model", run.info.run_id
    )
    return str(version.version)


def test_the_pin_includes_the_version(registered, spy, store, tmp_path):
    second = register_second_version_of_the_same_bytes(store, tmp_path)
    assert second == "2"
    pin_1 = vm.model_sha256(vm.download(ALIAS_URI, store, version=1))
    pin_2 = vm.model_sha256(vm.download(ALIAS_URI, store, version=2))
    assert pin_1 == registered.result["sha256"]
    assert pin_1 != pin_2

    def load(version, pin):
        return vm.verify_then_load(
            ALIAS_URI,
            expected_sha256=pin,
            tracking_uri=store,
            trusted_types={TREE},
            version=version,
        )

    assert load("1", pin_1)[1] == pin_1
    assert load("2", pin_2)[1] == pin_2
    with pytest.raises(vm.ModelVerificationError, match="mismatch"):
        load("2", pin_1)
    with pytest.raises(vm.ModelVerificationError, match="mismatch"):
        load("1", pin_2)


def test_model_sha256_still_covers_every_file(tmp_path):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "a.txt").write_bytes(b"one")
    (folder / "sub").mkdir()
    (folder / "sub" / "b.txt").write_bytes(b"two")
    base = vm.model_sha256(folder)
    assert vm.model_sha256(folder) == base

    (folder / "extra.txt").write_bytes(b"")
    added = vm.model_sha256(folder)
    assert added != base
    (folder / "extra.txt").unlink()
    assert vm.model_sha256(folder) == base

    (folder / "sub" / "b.txt").rename(folder / "sub" / "c.txt")
    assert vm.model_sha256(folder) != base
    (folder / "sub" / "c.txt").rename(folder / "sub" / "b.txt")

    (folder / "sub" / "b.txt").write_bytes(b"twp")
    assert vm.model_sha256(folder) != base
