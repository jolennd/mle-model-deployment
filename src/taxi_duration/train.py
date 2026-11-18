"""Baseline training: fold fingerprints, cross-validation and the lock record."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow import MlflowClient

from taxi_duration import config, features, splits
from taxi_duration.ingest import source_hash

TARGET = "duration_minutes"


def fingerprint(ids) -> str:
    """SHA-256 of the sorted ids joined with a newline."""
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode()).hexdigest()


def fold_fingerprints(folds: list[dict]) -> list[dict]:
    return [
        {
            "train_rows": len(fold["train"]),
            "valid_rows": len(fold["valid"]),
            "train_sha256": fingerprint(fold["train"]),
            "valid_sha256": fingerprint(fold["valid"]),
        }
        for fold in folds
    ]


def _rmse(actual, predicted) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def cross_validate(
    df: pd.DataFrame, folds: list[dict], pipeline_factory=features.make_pipeline
) -> dict:
    """RMSE in minutes per fold for the model and the training-mean reference.

    Every pipeline is fitted on its training fold only. The forest runs with
    n_jobs=-1 for speed; with a fixed random_state the trees are the same.
    """
    by_id = df.set_index("trip_id")
    model_scores, reference_scores = [], []
    for fold in folds:
        train = by_id.loc[fold["train"]]
        valid = by_id.loc[fold["valid"]]
        pipe = pipeline_factory()
        pipe.set_params(model__n_jobs=-1)
        pipe.fit(features.build_features(train), train[TARGET])
        predicted = pipe.predict(features.build_features(valid))
        model_scores.append(_rmse(valid[TARGET], predicted))
        reference_scores.append(_rmse(valid[TARGET], train[TARGET].mean()))
    return {"model": model_scores, "mean_reference": reference_scores}


def build_lock(
    cv: dict, folds: list[dict], sample: pd.DataFrame, source_sha256: str
) -> dict:
    return {
        "metric": "rmse_minutes",
        "fold_scores": [float(x) for x in cv["model"]],
        "mean_reference_scores": [float(x) for x in cv["mean_reference"]],
        "mean_rmse": float(np.mean(cv["model"])),
        "mean_reference_mean_rmse": float(np.mean(cv["mean_reference"])),
        "sample_rows": len(sample),
        "seed": config.SEED,
        "sample_sha256": fingerprint(sample["trip_id"]),
        "source_sha256": source_sha256,
        "folds": fold_fingerprints(folds),
    }


def register_if_allowed(size_bytes: int, register) -> bool:
    """Call register() only when the model file is strictly under the size limit."""
    if size_bytes < config.MODEL_SIZE_LIMIT_BYTES:
        register()
        return True
    return False


def _experiment_id(client: MlflowClient, tracking_uri: str) -> str:
    """Find or create the experiment; keep its artifact folder beside a sqlite store."""
    found = client.get_experiment_by_name(config.MODEL_NAME)
    if found is not None:
        return found.experiment_id
    location = None
    if tracking_uri.startswith("sqlite:///"):
        db_path = Path(tracking_uri[len("sqlite:///") :]).resolve()
        location = (db_path.parent / "mlflow-artifacts").as_uri()
    return client.create_experiment(config.MODEL_NAME, artifact_location=location)


def log_run(tracking_uri, cv, lock, size_bytes, params) -> str:
    """Log params, per-fold and mean scores and the file size. No model, no registry."""
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.create_run(_experiment_id(client, tracking_uri))
    run_id = run.info.run_id
    for key, value in params.items():
        client.log_param(run_id, key, value)
    for number, score in enumerate(cv["model"], start=1):
        client.log_metric(run_id, f"fold_{number}_rmse", float(score))
    client.log_metric(run_id, "mean_rmse", float(lock["mean_rmse"]))
    client.log_metric(
        run_id, "mean_reference_mean_rmse", float(lock["mean_reference_mean_rmse"])
    )
    if size_bytes is not None:
        client.log_metric(run_id, "model_size_bytes", float(size_bytes))
    client.set_terminated(run_id)
    return run_id


def _source_sha256() -> str:
    """Hash of the source file, from the hash record the load step also checks."""
    name = config.DOWNLOAD_FILE_PATTERN.format(partition=config.PARTITIONS[0])
    record = source_hash._project_root() / "tests" / "reference" / "source_sha256.txt"
    return source_hash._read_record(record)[name]


def run_baseline(
    sample_size=config.SAMPLE_SIZE, n_estimators=100, tracking_uri=None
) -> dict:
    """Cross-validate the baseline, write the lock, log the run. Never overwrites a lock."""
    lock_path = Path(config.LOCK_PATH)
    if lock_path.exists():
        raise FileExistsError(f"baseline lock already exists: {lock_path}")
    train = splits.load_train()
    sample = splits.draw_sample(train, size=sample_size)
    folds = splits.make_folds(sample)

    def factory():
        return features.make_pipeline().set_params(model__n_estimators=n_estimators)

    cv = cross_validate(sample, folds, pipeline_factory=factory)
    lock = build_lock(cv, folds, sample, _source_sha256())
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True))
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        params = {
            "n_estimators": n_estimators,
            "seed": config.SEED,
            "sample_rows": len(sample),
        }
        log_run(uri, cv, lock, None, params)
    return {
        "mean_rmse": lock["mean_rmse"],
        "mean_reference_mean_rmse": lock["mean_reference_mean_rmse"],
        "fold_scores": lock["fold_scores"],
        "lock_path": str(lock_path),
    }
