"""Fold fingerprints and the lock record: fixed rule, stable hashes, no test-week data."""

import hashlib
import json

import pytest

from taxi_duration import config


def expected_hash(ids):
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode()).hexdigest()


def two_folds():
    return [
        {"train": ["t3", "t1", "t2"], "valid": ["t5", "t4"]},
        {"train": ["t1", "t2", "t3", "t4", "t5"], "valid": ["t7", "t6", "t8"]},
    ]


def test_fold_fingerprints():
    from taxi_duration import train

    folds = two_folds()
    result = train.fold_fingerprints(folds)

    assert len(result) == len(folds)
    for fold, entry in zip(folds, result, strict=True):
        assert entry["train_rows"] == len(fold["train"])
        assert entry["valid_rows"] == len(fold["valid"])
        assert entry["train_sha256"] == expected_hash(fold["train"])
        assert entry["valid_sha256"] == expected_hash(fold["valid"])

    # fixed rule: sorted ids joined with newline
    assert train.fingerprint(["b", "a"]) == hashlib.sha256(b"a\nb").hexdigest()
    # order does not matter
    assert train.fingerprint(["c", "a", "b"]) == train.fingerprint(["a", "b", "c"])
    # one changed id changes the hash
    assert train.fingerprint(["a", "b", "c"]) != train.fingerprint(["a", "b", "d"])
    # different folds differ
    assert result[0]["valid_sha256"] != result[1]["valid_sha256"]
    assert result[0]["train_sha256"] != result[1]["train_sha256"]


def test_lock_has_required_fields():
    import pandas as pd

    from taxi_duration import train

    folds = two_folds()
    cv = {"model": [3.0, 2.5], "mean_reference": [5.0, 4.5]}
    sample = pd.DataFrame({"trip_id": ["t2", "t1", "t3"]})

    lock = train.build_lock(cv, folds, sample, "abc123")

    assert set(lock) == {
        "metric",
        "fold_scores",
        "mean_reference_scores",
        "mean_rmse",
        "mean_reference_mean_rmse",
        "sample_rows",
        "seed",
        "sample_sha256",
        "source_sha256",
        "folds",
    }
    assert lock["metric"] == "rmse_minutes"
    assert len(lock["fold_scores"]) == len(folds)
    assert len(lock["mean_reference_scores"]) == len(folds)
    assert lock["mean_rmse"] == pytest.approx(sum(cv["model"]) / 2)
    assert lock["mean_reference_mean_rmse"] == pytest.approx(4.75)
    assert isinstance(lock["mean_rmse"], float)
    assert lock["sample_rows"] == 3
    assert lock["seed"] == config.SEED
    assert lock["sample_sha256"] == expected_hash(["t1", "t2", "t3"])
    assert lock["source_sha256"] == "abc123"
    assert lock["folds"] == train.fold_fingerprints(folds)

    text = json.dumps(lock)
    assert "test" not in " ".join(lock).lower()
    assert "2025-01-25" not in text
