"""Reported metrics and the single test read (docs/tuning-plan.md).

evaluate_test never imports the test loader: the caller passes it in. Only
run_evaluation wires the real one, and only that function is the entry point.
"""

import json
import os
from pathlib import Path

import numpy as np

from taxi_duration import config, features

TARGET = "duration_minutes"
AIRPORT_ZONE_IDS = (1, 132, 138)
SEGMENTS = (
    "duration_1_10",
    "duration_10_30",
    "duration_30_60",
    "airport",
    "other",
)


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error**2)))


def _segment_masks(actual: np.ndarray, zones: np.ndarray) -> dict:
    airport = np.isin(zones, AIRPORT_ZONE_IDS)
    return {
        "duration_1_10": (actual >= 1) & (actual < 10),
        "duration_10_30": (actual >= 10) & (actual < 30),
        "duration_30_60": (actual >= 30) & (actual <= 60),
        "airport": airport,
        "other": ~airport,
    }


def report_metrics(actual, predicted, pickup_zone_id) -> dict:
    """Reported metrics (never used to choose). Error is predicted minus actual."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    zones = np.asarray(pickup_zone_id)
    error = predicted - actual
    ss_tot = float(np.sum((actual - actual.mean()) ** 2)) if len(actual) else 0.0
    r2 = float(1 - np.sum(error**2) / ss_tot) if ss_tot > 0 else None
    rmse_by_segment, rows_by_segment = {}, {}
    for name, mask in _segment_masks(actual, zones).items():
        count = int(mask.sum())
        rows_by_segment[name] = count
        rmse_by_segment[name] = _rmse(error[mask]) if count else None
    return {
        "rmse": _rmse(error) if len(actual) else None,
        "mae": float(np.mean(np.abs(error))) if len(actual) else None,
        "r2": r2,
        "median_ae": float(np.median(np.abs(error))) if len(actual) else None,
        "mean_error": float(np.mean(error)) if len(actual) else None,
        "rmse_by_segment": rmse_by_segment,
        "rows_by_segment": rows_by_segment,
        "rows": len(actual),
    }


def evaluate_test(model_loader, baseline_loader, *, load_test, results_dir) -> dict:
    """Score the chosen model and the baseline once on the test rows."""
    target = Path(results_dir) / "metrics.json"
    if target.exists():
        raise FileExistsError(f"metrics file already exists: {target}")
    frame = load_test()
    chosen_model = model_loader()
    baseline_model = baseline_loader()
    feats = features.build_features(frame)
    actual = frame[TARGET].to_numpy()
    zones = frame["pickup_zone_id"].to_numpy()
    result = {
        "chosen": report_metrics(
            actual, np.asarray(chosen_model.predict(feats)), zones
        ),
        "baseline": report_metrics(
            actual, np.asarray(baseline_model.predict(feats)), zones
        ),
        "rows_used": len(frame),
        "test_reads": 1,
    }
    if len(frame):
        result["pickup_range"] = [
            str(frame["pickup_at"].min()),
            str(frame["pickup_at"].max()),
        ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def _trusted_types() -> frozenset:
    raw = os.environ.get("MODEL_TRUSTED_TYPES", "")
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


def _loader(uri_name: str, sha_name: str, tracking_uri: str, trusted):
    def load():
        from taxi_duration import verify_model

        model, _ = verify_model.verify_then_load(
            config.require_env(uri_name),
            expected_sha256=config.require_env(sha_name),
            tracking_uri=tracking_uri,
            trusted_types=trusted,
        )
        return model

    return load


def run_evaluation(results_dir=config.RESULTS_DIR) -> dict:
    """Entry point: the one real test read. Not run by tests."""
    from taxi_duration import splits

    tracking_uri = config.require_env("MLFLOW_TRACKING_URI")
    config.require_env("MODEL_URI")
    config.require_env("MODEL_SHA256")
    for name in ("BASELINE_MODEL_URI", "BASELINE_MODEL_SHA256"):
        if not os.environ.get(name):
            raise ValueError(
                f"{name} is not set. The baseline is registered only if its file is "
                f"under {config.MODEL_SIZE_LIMIT_BYTES} bytes; if it is over the "
                "limit it is not in the store and cannot be scored here. "
                "Stopping instead of guessing."
            )
    trusted = _trusted_types()
    return evaluate_test(
        _loader("MODEL_URI", "MODEL_SHA256", tracking_uri, trusted),
        _loader("BASELINE_MODEL_URI", "BASELINE_MODEL_SHA256", tracking_uri, trusted),
        load_test=splits.load_test,
        results_dir=results_dir,
    )
