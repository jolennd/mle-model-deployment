"""train.run_baseline reads MLFLOW_TRACKING_URI strictly (ADR 8 point 19), review 3.

Seams (module attributes): splits.load_train, splits.draw_sample, splits.make_folds,
train.cross_validate, train.build_lock, train._source_sha256, train.log_run and
config.LOCK_PATH (a file in tmp_path). No data, no fit, no server.

Interface train.py must provide:
- When run_baseline is called without an explicit tracking_uri and the environment
  variable MLFLOW_TRACKING_URI is set, the value must pass
  verify_model.check_strict_tracking_uri (only http://127.0.0.1:5001, optional trailing
  slash). Otherwise run_baseline raises ValueError whose message names
  MLFLOW_TRACKING_URI, before load_train, any fit, the lock write or log_run.
- An explicit tracking_uri argument is not checked (a sqlite store stays usable).
- With the environment variable unset, nothing is logged and run_baseline proceeds as
  today.
"""

from types import SimpleNamespace

import pytest

GOOD = ["http://127.0.0.1:5001", "http://127.0.0.1:5001/"]
BAD = [
    "sqlite:///x.db",
    "http://localhost:5001",
    "https://127.0.0.1:5001",
    "http://127.0.0.1:5000",
    "http://127.0.0.1",
    "http://user@127.0.0.1:5001",
    "http://user" + ":pw@127.0.0.1:5001",
    "http://127.0.0.1:5001/api",
]


@pytest.fixture
def seams(monkeypatch, tmp_path):
    from taxi_duration import config, splits, train

    seen = SimpleNamespace(loaded=[], fits=[], logged=[], lock=tmp_path / "lock.json")
    monkeypatch.setattr(config, "LOCK_PATH", str(seen.lock))

    def load_train():
        seen.loaded.append(1)
        return "rows"

    def cross_validate(sample, folds, pipeline_factory=None):
        seen.fits.append(1)
        return {"model": [1.0, 2.0, 3.0], "mean_reference": [4.0, 5.0, 6.0]}

    def build_lock(cv, folds, sample, source_sha256):
        return {
            "mean_rmse": 2.0,
            "mean_reference_mean_rmse": 5.0,
            "fold_scores": [1.0, 2.0, 3.0],
        }

    def log_run(uri, cv, lock, size_bytes, params):
        seen.logged.append(uri)
        return "run"

    monkeypatch.setattr(splits, "load_train", load_train)
    monkeypatch.setattr(splits, "draw_sample", lambda frame, size=None: [1, 2, 3])
    monkeypatch.setattr(splits, "make_folds", lambda sample: [])
    monkeypatch.setattr(train, "cross_validate", cross_validate)
    monkeypatch.setattr(train, "build_lock", build_lock)
    monkeypatch.setattr(train, "_source_sha256", lambda: "sha")
    monkeypatch.setattr(train, "log_run", log_run)
    return seen


@pytest.mark.parametrize("uri", BAD)
def test_bad_environment_uri_is_refused_before_any_work(monkeypatch, seams, uri):
    from taxi_duration import train

    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        train.run_baseline(sample_size=10, n_estimators=2)

    assert seams.loaded == []
    assert seams.fits == []
    assert seams.logged == []
    assert not seams.lock.exists()


@pytest.mark.parametrize("uri", GOOD)
def test_good_environment_uri_proceeds_and_logs(monkeypatch, seams, uri):
    from taxi_duration import train

    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    result = train.run_baseline(sample_size=10, n_estimators=2)

    assert result["mean_rmse"] == 2.0
    assert seams.fits == [1]
    assert seams.lock.exists()
    assert seams.logged == [uri]


def test_explicit_tracking_uri_is_not_checked(monkeypatch, seams):
    from taxi_duration import train

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    train.run_baseline(sample_size=10, n_estimators=2, tracking_uri="sqlite:///x.db")

    assert seams.logged == ["sqlite:///x.db"]


def test_unset_environment_uri_logs_nothing(monkeypatch, seams):
    from taxi_duration import train

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    train.run_baseline(sample_size=10, n_estimators=2)

    assert seams.fits == [1]
    assert seams.logged == []
