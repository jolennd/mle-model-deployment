"""evaluate_test order, extra metrics content and the baseline refit speed (batch 4).

Interface evaluate.py must provide (old calls stay valid):
- evaluate_test(model_loader, baseline_loader, *, load_test, results_dir,
  train_mean=None, model_info=None, cv=None) -> dict.
  Calls model_loader() then baseline_loader() BEFORE load_test(). If a loader raises,
  load_test is never called and no metrics.json is written.
  train_mean (float): adds "mean_reference" = report_metrics of a constant prediction
  equal to train_mean on the test rows (same keys as "chosen").
  model_info (dict): stored as "model_info". cv (dict): stored as "cv".
  All three appear in the returned dict and in metrics.json.
- cv_reports(sample, folds, factories) -> {name: [report per fold, ...] for each name
  in factories, plus "mean_reference": [...]}. Per fold: factories[name]() is fitted
  on the fold's train rows (by trip_id, features.build_features, duration_minutes),
  reported with report_metrics on the valid rows. The mean reference predicts the
  fold's training mean.
- refit_baseline fits with n_jobs=-1 and sets n_jobs=1 on the forest before returning.
"""

import json

import numpy as np
import pandas as pd
import pytest

TARGET = "duration_minutes"


def make_test_frame():
    pickup = pd.date_range("2025-01-25 08:00", periods=6, freq="h")
    return pd.DataFrame(
        {
            "trip_id": [f"t{i}" for i in range(6)],
            "pickup_at": pickup,
            "dropoff_at": pickup + pd.Timedelta(minutes=15),
            "pickup_zone_id": [132, 10, 10, 20, 20, 138],
            "dropoff_zone_id": [30, 30, 40, 40, 30, 40],
            "trip_distance": [10.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "duration_minutes": [30.0, 5.0, 10.0, 15.0, 20.0, 40.0],
        }
    )


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


def test_loaders_run_before_the_test_rows_are_read(tmp_path):
    from taxi_duration import evaluate

    order = []

    def load_test():
        order.append("load_test")
        return make_test_frame()

    def model_loader():
        order.append("model")
        return Constant(20.0)

    def baseline_loader():
        order.append("baseline")
        return Constant(10.0)

    evaluate.evaluate_test(
        model_loader, baseline_loader, load_test=load_test, results_dir=tmp_path
    )

    assert order == ["model", "baseline", "load_test"]


@pytest.mark.parametrize("failing", ["model", "baseline"])
def test_a_failing_loader_never_opens_the_test_rows(tmp_path, failing):
    from taxi_duration import evaluate

    reads = []

    def load_test():
        reads.append(1)
        return make_test_frame()

    def loader(name):
        def load():
            if name == failing:
                raise RuntimeError(f"{name} could not be loaded")
            return Constant(1.0)

        return load

    results = tmp_path / "results"
    with pytest.raises(RuntimeError, match="could not be loaded"):
        evaluate.evaluate_test(
            loader("model"),
            loader("baseline"),
            load_test=load_test,
            results_dir=results,
        )

    assert reads == []
    assert not (results / "metrics.json").exists()


def test_metrics_hold_mean_reference_model_info_and_cv(tmp_path):
    from taxi_duration import evaluate

    frame = make_test_frame()
    actual = frame[TARGET].to_numpy()
    info = {"alias": "champion", "version": "3", "sha256": "ab" * 32}
    cv = {"chosen": [{"rmse": 1.0}], "baseline": [{"rmse": 2.0}]}

    got = evaluate.evaluate_test(
        lambda: Constant(20.0),
        lambda: Constant(10.0),
        load_test=lambda: frame,
        results_dir=tmp_path,
        train_mean=12.0,
        model_info=info,
        cv=cv,
    )

    reference = got["mean_reference"]
    assert set(reference) == set(got["chosen"])  # same report_metrics keys
    assert reference["rmse"] == pytest.approx(np.sqrt(np.mean((12.0 - actual) ** 2)))
    assert reference["mean_error"] == pytest.approx(12.0 - actual.mean())
    assert reference["rows"] == 6
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["mean_reference"]["rmse"] == pytest.approx(reference["rmse"])
    assert saved["model_info"] == info
    assert saved["cv"] == cv
    assert got["model_info"] == info
    assert got["cv"] == cv
    assert saved["test_reads"] == 1


class Probe:
    """Estimator stand-in: records the train rows it was fitted on."""

    def __init__(self, value, fits):
        self.value = value
        self.fits = fits
        self.named_steps = {"model": self}

    def set_params(self, **kwargs):
        return self

    def fit(self, X, y):
        self.fits.append((list(X.index), list(y)))
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def make_sample(n=60):
    rng = np.random.default_rng(5)
    distance = rng.uniform(0.5, 12, n)
    return pd.DataFrame(
        {
            "trip_id": [f"t{i:03d}" for i in range(n)],
            "pickup_at": pd.Timestamp("2025-01-05")
            + pd.to_timedelta(np.arange(n) * 30, unit="m"),
            "pickup_zone_id": rng.choice([10, 132], n),
            "dropoff_zone_id": rng.choice([30, 40], n),
            "trip_distance": distance,
            "duration_minutes": 2 + 4 * distance,
        }
    )


def test_cv_reports_fit_each_fold_and_report_on_valid_rows():
    from taxi_duration import evaluate

    sample = make_sample()
    ids = sample["trip_id"].tolist()
    folds = [
        {"train": ids[:20], "valid": ids[20:30]},
        {"train": ids[:30], "valid": ids[30:40]},
        {"train": ids[:40], "valid": ids[40:50]},
    ]
    chosen_fits, base_fits = [], []

    got = evaluate.cv_reports(
        sample,
        folds,
        {
            "chosen": lambda: Probe(15.0, chosen_fits),
            "baseline": lambda: Probe(25.0, base_fits),
        },
    )

    assert set(got) == {"chosen", "baseline", "mean_reference"}
    by_id = sample.set_index("trip_id")
    assert [f[0] for f in chosen_fits] == [fold["train"] for fold in folds]
    assert [f[0] for f in base_fits] == [fold["train"] for fold in folds]
    for name, constant in (("chosen", 15.0), ("baseline", 25.0)):
        assert len(got[name]) == 3
    for number, fold in enumerate(folds):
        valid = by_id.loc[fold["valid"]]
        actual = valid[TARGET].to_numpy()
        train_mean = by_id.loc[fold["train"]][TARGET].mean()
        for name, constant in (
            ("chosen", 15.0),
            ("baseline", 25.0),
            ("mean_reference", train_mean),
        ):
            report = got[name][number]
            assert report["rows"] == len(valid)
            assert report["rmse"] == pytest.approx(
                np.sqrt(np.mean((constant - actual) ** 2))
            )
        assert set(got["mean_reference"][number]) == set(got["chosen"][number])


class RecordingForest:
    def __init__(self):
        self.n_jobs = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self


class RecordingPipeline:
    def __init__(self, seen):
        self.forest = RecordingForest()
        self.named_steps = {"model": self.forest}
        self.seen = seen

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key == "model__n_jobs":
                self.forest.n_jobs = value
        return self

    def fit(self, X, y):
        self.seen.append(self.forest.n_jobs)
        return self


def test_refit_baseline_fits_with_all_cores_then_sets_one_job():
    from taxi_duration import evaluate, train

    sample = make_sample()
    lock = {"sample_sha256": train.fingerprint(sample["trip_id"])}
    seen, built = [], []

    def factory():
        built.append(RecordingPipeline(seen))
        return built[-1]

    fitted = evaluate.refit_baseline(sample, lock, pipeline_factory=factory)

    assert seen == [-1]  # n_jobs at fit time
    assert fitted.named_steps["model"].n_jobs == 1  # before it is returned
