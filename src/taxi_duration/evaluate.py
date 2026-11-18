"""Reported metrics and the single test read (docs/tuning-plan.md).

evaluate_test never imports the test loader: the caller passes it in. Only
run_evaluation wires the real one, and only that function is the entry point.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from taxi_duration import config, features, train

TARGET = "duration_minutes"
AIRPORT_ZONE_IDS = (1, 132, 138)
MARKER = "test_read_started.json"
CHAMPION_URI = f"models:/{config.MODEL_NAME}@champion"
SEGMENTS = (
    "duration_1_10",
    "duration_10_30",
    "duration_30_60",
    "airport",
    "other",
)


def _guard(results_dir) -> None:
    """Refuse a second test read: metrics.json, or a marker left by a failed attempt."""
    folder = Path(results_dir)
    if (folder / "metrics.json").exists():
        raise FileExistsError(f"metrics file already exists: {folder / 'metrics.json'}")
    if (folder / MARKER).exists():
        raise RuntimeError(
            f"{MARKER} exists without metrics.json: an earlier attempt may have read "
            "the test week and failed before writing results. Stopping. Removing the "
            "marker needs an ADR that says the test week was read more than once "
            "(ADR 8 point 18)."
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


def evaluate_test(
    model_loader,
    baseline_loader,
    *,
    load_test,
    results_dir,
    train_mean=None,
    model_info=None,
    cv=None,
) -> dict:
    """Score the chosen model and the baseline once on the test rows.

    Both loaders run before the test rows are opened, so a failing loader
    never reads them and writes no file. The marker is written just before
    load_test and stays (ADR 8 point 18).
    """
    _guard(results_dir)
    chosen_model = model_loader()
    baseline_model = baseline_loader()
    marker = Path(results_dir) / MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"started_at": datetime.now().isoformat()}))
    frame = load_test()
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
    if train_mean is not None:
        result["mean_reference"] = report_metrics(
            actual, np.full(len(actual), float(train_mean)), zones
        )
    if model_info is not None:
        result["model_info"] = model_info
    if cv is not None:
        result["cv"] = cv
    if len(frame):
        result["pickup_range"] = [
            str(frame["pickup_at"].min()),
            str(frame["pickup_at"].max()),
        ]
    (Path(results_dir) / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    return result


def cv_reports(sample, folds, factories) -> dict:
    """Reported metrics per fold on the valid rows; fits on the train rows only."""
    by_id = sample.set_index("trip_id")
    out = {name: [] for name in factories}
    out["mean_reference"] = []
    for fold in folds:
        train_rows = by_id.loc[fold["train"]]
        valid = by_id.loc[fold["valid"]]
        train_feats = features.build_features(train_rows)
        valid_feats = features.build_features(valid)
        actual = valid[TARGET].to_numpy()
        zones = valid["pickup_zone_id"].to_numpy()
        for name, factory in factories.items():
            pipe = factory()
            pipe.set_params(model__n_jobs=-1)
            pipe.fit(train_feats, train_rows[TARGET])
            out[name].append(
                report_metrics(actual, np.asarray(pipe.predict(valid_feats)), zones)
            )
        constant = np.full(len(actual), float(train_rows[TARGET].mean()))
        out["mean_reference"].append(report_metrics(actual, constant, zones))
    return out


def refit_baseline(sample, lock, *, pipeline_factory=None):
    """Refit the baseline in memory (ADR 8 points 15 and 16). Saves and loads no file."""
    if train.fingerprint(sample["trip_id"]) != lock["sample_sha256"]:
        raise ValueError(
            "sample_sha256 does not match the lock: the sample is not the one the "
            "baseline was locked on. Stopping."
        )
    factory = pipeline_factory or features.make_pipeline
    pipeline = factory()
    pipeline.set_params(model__n_jobs=-1)
    pipeline.fit(features.build_features(sample), sample[TARGET])
    # n_jobs=-1 only speeds the fit; one job matches single-row serving
    pipeline.named_steps["model"].set_params(n_jobs=1)
    return pipeline


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


def _chosen_params(results_dir) -> dict:
    """Final model settings from trials.json. Loud when missing or unusable."""
    path = Path(results_dir) / "trials.json"
    if not path.is_file():
        raise FileNotFoundError(f"trials.json not found in {results_dir}")
    try:
        params = json.loads(path.read_text())["final"]["params"]
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"trials.json has no usable final.params: {error!r}"
        ) from error
    if not isinstance(params, dict):
        raise ValueError("trials.json final.params is not a dict")  # noqa: TRY004
    return dict(params)


def _factory(params):
    def build():
        return features.make_pipeline().set_params(
            **{f"model__{k}": v for k, v in params.items()}
        )

    return build


def run_evaluation(results_dir=config.RESULTS_DIR) -> dict:
    """Entry point: the one real test read. Not run by tests."""
    from mlflow import MlflowClient

    from taxi_duration import splits, verify_model

    # 1. strict settings
    tracking_uri = verify_model.check_strict_tracking_uri(
        config.require_env("MLFLOW_TRACKING_URI")
    )
    model_uri = config.require_env("MODEL_URI")
    if model_uri != CHAMPION_URI:
        raise ValueError(f"MODEL_URI must be exactly {CHAMPION_URI}, got {model_uri!r}")
    expected_sha = config.require_env("MODEL_SHA256")
    trusted = _trusted_types()

    # 2. guard, before anything is downloaded or fitted
    _guard(results_dir)
    params = _chosen_params(results_dir)

    # 3. one alias lookup, then one pinned download and verification
    alias = model_uri[len("models:/") :].split("@", 1)[1]
    version = str(
        MlflowClient(tracking_uri=tracking_uri)
        .get_model_version_by_alias(config.MODEL_NAME, alias)
        .version
    )
    champion, sha = verify_model.verify_then_load(
        model_uri,
        expected_sha256=expected_sha,
        tracking_uri=tracking_uri,
        trusted_types=trusted,
        version=version,
    )
    model_info = {"alias": alias, "version": version, "sha256": sha}

    # 4. the long compute. Training rows only: the test week is not touched.
    sample = splits.draw_sample(splits.load_train())
    lock = train.read_lock()
    folds = splits.make_folds(sample)
    train_mean = float(sample[TARGET].mean())
    cv = cv_reports(
        sample, folds, {"baseline": features.make_pipeline, "chosen": _factory(params)}
    )

    return evaluate_test(
        lambda: champion,
        lambda: refit_baseline(sample, lock),
        load_test=splits.load_test,
        results_dir=results_dir,
        train_mean=train_mean,
        model_info=model_info,
        cv=cv,
    )
