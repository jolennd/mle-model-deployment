"""Baseline training: fold fingerprints, cross-validation and the lock record."""

import hashlib

import numpy as np
import pandas as pd

from taxi_duration import config, features

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
