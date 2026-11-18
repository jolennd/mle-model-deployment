"""One run is logged to the model store: params, scores and size, no model file."""

import pytest
from mlflow import MlflowClient

pytestmark = pytest.mark.integration


def test_run_logged_without_model_artifact(tmp_path):
    from taxi_duration import train

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cv = {"model": [4.0, 5.0, 6.0], "mean_reference": [8.0, 9.0, 10.0]}
    lock = {
        "fold_scores": cv["model"],
        "mean_reference_scores": cv["mean_reference"],
        "mean_rmse": 5.0,
        "mean_reference_mean_rmse": 9.0,
    }
    params = {"n_estimators": 5, "seed": 42, "sample_rows": 400}

    run_id = train.log_run(uri, cv, lock, 12345, params)

    client = MlflowClient(tracking_uri=uri)
    experiments = [e.experiment_id for e in client.search_experiments()]
    runs = client.search_runs(experiments)
    assert [r.info.run_id for r in runs] == [run_id]

    run = client.get_run(run_id)
    assert run.data.params == {k: str(v) for k, v in params.items()}
    metrics = run.data.metrics
    assert metrics["fold_1_rmse"] == pytest.approx(4.0)
    assert metrics["fold_2_rmse"] == pytest.approx(5.0)
    assert metrics["fold_3_rmse"] == pytest.approx(6.0)
    assert metrics["mean_rmse"] == pytest.approx(5.0)
    assert metrics["mean_reference_mean_rmse"] == pytest.approx(9.0)
    assert metrics["model_size_bytes"] == 12345

    assert client.list_artifacts(run_id) == []
    assert list(client.search_logged_models(experiment_ids=experiments)) == []
    assert client.search_registered_models() == []
