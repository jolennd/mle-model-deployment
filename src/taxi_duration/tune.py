"""Tuning exactly as pre-registered in docs/tuning-plan.md.

Budget, search space, folds, metric and decision rule come from that file and
ADR 8. The pure parts (search space, choice, limits, fallback) are tested; the
run_tuning and register_champion functions at the bottom are thin glue.
"""

import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import optuna

from taxi_duration import artifacts, config, features, splits, train
from taxi_duration.train import TARGET

MAX_TRIALS = 30
MAX_SECONDS = 45 * 60
MIN_GAIN = 0.02
DEPTH_CAPS = (25, 20, 15, 12, 10, 8)
MIN_TREES = 25
DEFAULT_TREES = 100  # scikit-learn default, used for the baseline ({})
MODEL_DIR = "models/champion_candidate"


def suggest_params(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 8, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", 0.3, 0.5, 1.0]
        ),
        "max_samples": trial.suggest_float("max_samples", 0.3, 1.0),
    }


def make_study():
    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
        pruner=optuna.pruners.NopPruner(),
    )


def run_study(study, objective, n_trials=MAX_TRIALS, timeout=MAX_SECONDS):
    study.optimize(objective, n_trials=n_trials, timeout=timeout)


def _factory(params):
    def build():
        return features.make_pipeline().set_params(
            **{f"model__{k}": v for k, v in params.items()}
        )

    return build


def score_params(sample, params) -> list[float]:
    folds = splits.make_folds(sample)
    cv = train.cross_validate(sample, folds, pipeline_factory=_factory(params))
    return [float(x) for x in cv["model"]]


def _mean(record) -> float:
    return statistics.fmean(record["fold_rmse"])


def choose(baseline, candidates):
    """Decision rule steps 1 and 2 (limits are applied by choose_final)."""
    base_mean = _mean(baseline)
    eligible = [
        c
        for c in candidates
        if (base_mean - _mean(c)) / base_mean >= MIN_GAIN
        and all(a < b for a, b in zip(c["fold_rmse"], baseline["fold_rmse"]))
    ]
    if eligible:
        return min(eligible, key=_mean)
    return baseline


def passes_limits(size_bytes, predict_ms) -> bool:
    return size_bytes < config.MODEL_SIZE_LIMIT_BYTES and predict_ms < (
        config.MAX_PREDICT_MS
    )


def _passes(record) -> bool:
    return passes_limits(record["size_bytes"], record["predict_ms"])


def measure_predict_ms(fitted, row) -> float:
    fitted.named_steps["model"].set_params(n_jobs=1)
    fitted.predict(row)
    times = []
    for _ in range(100):
        start = time.perf_counter()
        fitted.predict(row)
        times.append((time.perf_counter() - start) * 1000)
    return float(statistics.median(times))


def choose_final(baseline, candidates):
    """Decision rule steps 1 to 3. None when nothing passes both limits."""
    passing = [c for c in candidates if _passes(c)]
    base_mean = _mean(baseline)
    eligible = [
        c
        for c in passing
        if (base_mean - _mean(c)) / base_mean >= MIN_GAIN
        and all(x < y for x, y in zip(c["fold_rmse"], baseline["fold_rmse"]))
    ]
    if eligible:
        return min(eligible, key=_mean)
    if _passes(baseline):
        return baseline
    if passing:
        return min(passing, key=_mean)
    return None


def skipped_caps(params) -> list[int]:
    depth = params.get("max_depth")
    if depth is None:
        return []
    return [cap for cap in DEPTH_CAPS if cap >= depth]


def fallback_attempts(params) -> list[dict]:
    skipped = skipped_caps(params)
    params = {"n_estimators": DEFAULT_TREES, **params}
    attempts = [{**params, "max_depth": c} for c in DEPTH_CAPS if c not in skipped]
    depth = params.get("max_depth")
    depth = DEPTH_CAPS[-1] if depth is None else min(depth, DEPTH_CAPS[-1])
    trees = params["n_estimators"]
    while trees > MIN_TREES:
        trees = max(trees // 2, MIN_TREES)
        attempts.append({**params, "max_depth": depth, "n_estimators": trees})
    return attempts


def select_fallback(params, check):
    """First attempt that passes both limits, else the last one with misses=True."""
    attempts = fallback_attempts(params) or [dict(params)]
    for attempt in attempts:
        if passes_limits(*check(attempt)):
            return attempt, False
    return attempts[-1], True


def first_passing_after_retrain(ordered_candidates, retrain_check):
    for candidate in ordered_candidates:
        if passes_limits(*retrain_check(candidate)):
            return candidate
    return None


# Glue below: needs real data, not unit tested, run only by the tuning step.


class NeedsApproval(RuntimeError):
    """The run stops for the human; nothing is registered."""


def _check_uri(uri: str) -> str:
    """Only an http server on 127.0.0.1, no user info."""
    parsed = urlparse(uri)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or "@" in parsed.netloc
    ):
        raise ValueError(f"MLFLOW_TRACKING_URI not allowed: {uri}")
    return uri


def prepare_model_dir(path) -> None:
    """Empty the folder (create it if missing); siblings are never touched."""
    folder = Path(path)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def _trusted_types() -> list[str]:
    raw = config.require_env("MODEL_TRUSTED_TYPES")
    return [t.strip() for t in raw.split(",") if t.strip()]


def _fit(rows, params):
    pipe = _factory(params)()
    pipe.set_params(model__n_jobs=-1)
    pipe.fit(features.build_features(rows), rows[TARGET])
    return pipe


def _size_and_speed(fitted, row, folder) -> tuple[int, float]:
    """Clear the folder, save (sets n_jobs=1), reload with trusted types, time rows."""
    prepare_model_dir(folder)
    size = artifacts.save_model(fitted, folder)
    loaded = artifacts.load_model(folder, _trusted_types())
    return size, measure_predict_ms(loaded, row)


def _fold3_measure(sample, folds, params) -> tuple[int, float]:
    by_id = sample.set_index("trip_id")
    fit_rows = by_id.loc[folds[-1]["train"]]
    row = features.build_features(by_id.loc[folds[-1]["valid"]].iloc[[0]])
    fitted = _fit(fit_rows, params)
    with tempfile.TemporaryDirectory() as tmp:
        return _size_and_speed(fitted, row, Path(tmp) / "model")


BASELINE_SIZE_BYTES = 2_694_430_930  # measured in the training step (ADR 8 point 15)
BASELINE_SIZE_NOTE = (
    "baseline size from the training step skops file (ADR 8 point 15); "
    "baseline never saved again, speed measured in memory on the fold 3 refit"
)


def baseline_record(lock, predict_ms) -> dict:
    return {
        "name": "baseline",
        "fold_rmse": lock["fold_scores"],
        "size_bytes": BASELINE_SIZE_BYTES,
        "predict_ms": predict_ms,
    }


def measure_baseline_ms(sample, folds) -> float:
    """Fit the fold 3 baseline in memory and time single-row predictions."""
    by_id = sample.set_index("trip_id")
    fitted = _fit(by_id.loc[folds[-1]["train"]], {})
    row = features.build_features(by_id.loc[folds[-1]["valid"]].iloc[[0]])
    return measure_predict_ms(fitted, row)


def _log_record(client, experiment_id, record):
    run_id = client.create_run(experiment_id).info.run_id
    client.set_tag(run_id, "mlflow.runName", record["name"])
    for key, value in record["params"].items():
        client.log_param(run_id, key, value)
    for number, score in enumerate(record["fold_rmse"], start=1):
        client.log_metric(run_id, f"fold_{number}_rmse", score)
    client.log_metric(run_id, "mean_rmse", _mean(record))
    client.log_metric(run_id, "model_size_bytes", float(record["size_bytes"]))
    client.log_metric(run_id, "predict_ms", record["predict_ms"])
    client.set_terminated(run_id)


def _run_fallback(sample, row, candidate) -> dict:
    """Plan fallback: capped retrains, each saved, reloaded, timed and scored."""
    tried = []

    def check(attempt):
        fitted = _fit(sample, attempt)
        size, ms = _size_and_speed(fitted, row, MODEL_DIR)
        scores = score_params(sample, attempt)
        tried.append(
            {
                "params": attempt,
                "size_bytes": size,
                "predict_ms": ms,
                "mean_cv_rmse": statistics.fmean(scores),
            }
        )
        return size, ms

    _, misses = select_fallback(candidate["params"], check)
    return {
        "from": candidate["name"],
        "attempts": tried,
        "skipped_caps": skipped_caps(candidate["params"]),
        "capped_cv_rmse": tried[-1]["mean_cv_rmse"],
        "uncapped_cv_rmse": _mean(candidate),
        "misses": misses,
    }


def run_tuning(tracking_uri=None) -> dict:
    sample = splits.draw_sample(splits.load_train())
    folds = splits.make_folds(sample)
    lock = train.read_lock()
    if train.fingerprint(sample["trip_id"]) != lock["sample_sha256"]:
        raise ValueError("sample_sha256 differs from the baseline lock")
    if train.fold_fingerprints(folds) != lock["folds"]:
        raise ValueError("folds differ from the baseline lock")
    out = Path(config.RESULTS_DIR) / "trials.json"
    if out.exists():
        raise FileExistsError(f"tuning results already exist: {out}")

    from mlflow import MlflowClient

    uri = tracking_uri or _check_uri(config.require_env("MLFLOW_TRACKING_URI"))
    client = MlflowClient(tracking_uri=uri)
    experiment_id = train._experiment_id(client, uri)

    ms = measure_baseline_ms(sample, folds)
    baseline = baseline_record(lock, ms)
    baseline["params"] = {}
    baseline["fold_rmse"] = [float(x) for x in baseline["fold_rmse"]]
    records = []

    def write(extra):
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": baseline,
            "baseline_size_note": BASELINE_SIZE_NOTE,
            "trials": records,
            **extra,
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def objective(trial):
        params = suggest_params(trial)
        size, ms = _fold3_measure(sample, folds, params)
        record = {
            "name": f"trial_{trial.number}",
            "params": params,
            "fold_rmse": score_params(sample, params),
            "size_bytes": size,
            "predict_ms": ms,
        }
        records.append(record)
        _log_record(client, experiment_id, record)
        write({})
        return _mean(record)

    study = make_study()
    run_study(study, objective)
    chosen = choose_final(baseline, records)
    row = features.build_features(sample.iloc[[0]])

    if chosen is None:
        best = min([baseline, *records], key=_mean)
        fallback = _run_fallback(sample, row, best)
        write({"chosen": None, "fallback": fallback})
        message = (
            "No candidate passes the size and speed limits. The fallback ran. "
            "Nothing is registered; a human must read trials.json and approve."
        )
        if fallback["misses"]:
            message += " No capped model passes: the limits are missed."
        raise NeedsApproval(message)

    others = sorted(
        (c for c in [baseline, *records] if c is not chosen and _passes(c)), key=_mean
    )
    ordered = [chosen, *others]
    retrained = {}

    def retrain_check(candidate):
        fitted = _fit(sample, candidate["params"])
        size, ms = _size_and_speed(fitted, row, MODEL_DIR)
        retrained[candidate["name"]] = (size, ms)
        return size, ms

    final = first_passing_after_retrain(ordered, retrain_check)
    if final is None:
        write({"chosen": chosen["name"], "final": None})
        raise NeedsApproval(
            "No candidate passes the limits after retraining. Nothing is "
            "registered; a human must decide."
        )
    size, ms = retrained[final["name"]]
    # The loop stops at the first pass, so MODEL_DIR holds the final model.
    result = {
        "name": final["name"],
        "params": final["params"],
        "size_bytes": size,
        "predict_ms": ms,
        "model_dir": MODEL_DIR,
    }
    write({"chosen": chosen["name"], "final": result})
    return result


def _mlmodel_text(trusted_types) -> str:
    import yaml

    meta = {
        "flavors": {
            "python_function": {
                "loader_module": "mlflow.sklearn",
                "model_path": artifacts.MODEL_FILE,
                "predict_fn": "predict",
            },
            "sklearn": {
                "code": None,
                "pickled_model": artifacts.MODEL_FILE,
                "serialization_format": "skops",
                "skops_trusted_types": list(trusted_types),
            },
        },
    }
    return yaml.safe_dump(meta, sort_keys=False)


def register_champion(model_dir, size_bytes, predict_ms, tracking_uri=None) -> dict:
    """Register the saved model file as taxi-duration@champion; print its SHA-256."""
    import mlflow
    from mlflow import MlflowClient

    from taxi_duration import verify_model

    if not passes_limits(size_bytes, predict_ms):
        raise ValueError("model fails the size or speed limit; not registering")
    uri = tracking_uri or _check_uri(config.require_env("MLFLOW_TRACKING_URI"))
    trusted = _trusted_types()
    model_file = Path(model_dir) / artifacts.MODEL_FILE
    client = MlflowClient(tracking_uri=uri)
    experiment_id = train._experiment_id(client, uri)
    mlflow.set_tracking_uri(uri)
    with tempfile.TemporaryDirectory() as tmp:
        meta = Path(tmp) / "MLmodel"
        meta.write_text(_mlmodel_text(trusted))
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # The saved file is logged as is: no re-dump, no pickle.
            mlflow.log_artifact(str(model_file), artifact_path="model")
            mlflow.log_artifact(str(meta), artifact_path="model")
        try:
            client.create_registered_model(config.MODEL_NAME)
        except mlflow.exceptions.MlflowException:
            pass  # already registered: add a version
        version = client.create_model_version(
            config.MODEL_NAME, f"{run.info.artifact_uri}/model", run.info.run_id
        )
    client.set_registered_model_alias(config.MODEL_NAME, "champion", version.version)
    local = verify_model.download(f"models:/{config.MODEL_NAME}@champion", uri)
    sha = verify_model.model_sha256(local)
    print(f"MODEL_SHA256={sha}")
    return {
        "name": config.MODEL_NAME,
        "alias": "champion",
        "version": str(version.version),
        "sha256": sha,
    }


def register_from_results(tracking_uri=None) -> dict:
    """Register what run_tuning chose, read from trials.json."""
    path = Path(config.RESULTS_DIR) / "trials.json"
    data = json.loads(path.read_text())
    if data.get("fallback"):
        raise ValueError("the fallback ran: a human approval is needed first")
    final = data.get("final")
    if not final:
        raise ValueError("no model passed the limits; nothing to register")
    return register_champion(
        final["model_dir"],
        final["size_bytes"],
        final["predict_ms"],
        tracking_uri=tracking_uri,
    )
