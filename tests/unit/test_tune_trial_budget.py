"""Trial compute and disk (review 2, finding 3). Seams: tune_run_support.py.

Interface tune.py must provide:
- One fold-3 fit per trial: the trial fits the fold-3 train rows once, with
  tune._fit(rows, params), and uses that fitted model for the fold-3 score AND for
  the size/speed measurement. Folds 1 and 2 may still go through
  train.cross_validate, but fold 3 must not (the folds passed to it exclude it).
- A trial whose artifacts.estimate_size(fitted) is at or over
  2 * config.MODEL_SIZE_LIMIT_BYTES is not saved or reloaded (no
  artifacts.save_model, no artifacts.load_model). Its record has
  size_bytes == the estimate, predict_ms None and fold_rmse still set.
- passes_limits(size_bytes, None) is False; choose_final ignores such records and
  does not raise; the MLflow log skips the None predict_ms.
- A RuntimeError from artifacts.save_model (which calls check_disk) inside a trial is
  caught for that trial only: the record has "disk_error": true, predict_ms None,
  passes_limits(size_bytes, predict_ms) is False, the study continues and trials.json
  is still written after every trial. Other errors in a trial still propagate.
"""

import pytest
from tune_run_support import SMALL, params, read_trials, setup_run

from taxi_duration import config

GOOD = params(depth=12)
DEEP = params(depth=30)
LIMIT = config.MODEL_SIZE_LIMIT_BYTES
FOLD3_TRAIN_ROWS = 220  # tune_run_support.make_folds


def is_deep(p):
    return p["max_depth"] > 12


def wrap_saves(monkeypatch):
    from taxi_duration import artifacts

    saved, loaded = [], []
    real_save, real_load = artifacts.save_model, artifacts.load_model

    def save(fitted, folder):
        saved.append(dict(fitted.params))
        return real_save(fitted, folder)

    def load(folder, trusted):
        model = real_load(folder, trusted)
        loaded.append(dict(model.params))
        return model

    monkeypatch.setattr(artifacts, "save_model", save)
    monkeypatch.setattr(artifacts, "load_model", load)
    return saved, loaded


def test_one_fold3_fit_per_trial(monkeypatch, tmp_path):
    from taxi_duration import train, tune

    real_score = tune.score_params  # the real one, before setup_run stubs it
    seen = setup_run(monkeypatch, tmp_path, trials=[GOOD])
    monkeypatch.setattr(tune, "score_params", real_score)

    fold3_fits = []
    stub_fit = tune._fit

    def counting_fit(rows, fit_params):
        if len(rows) == FOLD3_TRAIN_ROWS:
            fold3_fits.append("tune._fit")
        return stub_fit(rows, fit_params)

    def cross_validate(df, folds, pipeline_factory):
        for fold in folds:
            if len(fold["train"]) == FOLD3_TRAIN_ROWS:
                fold3_fits.append("train.cross_validate")
        return {"model": [3.0] * len(folds), "mean_reference": [9.0] * len(folds)}

    monkeypatch.setattr(tune, "_fit", counting_fit)
    monkeypatch.setattr(train, "cross_validate", cross_validate)

    tune.run_tuning()

    assert fold3_fits == ["tune._fit"], fold3_fits
    trial = read_trials(seen)["trials"][0]
    assert len(trial["fold_rmse"]) == 3


@pytest.mark.parametrize(
    ("estimate", "kept"),
    [(2 * LIMIT - 1, True), (2 * LIMIT, False), (3 * LIMIT, False)],
)
def test_huge_estimate_skips_save_and_reload(monkeypatch, tmp_path, estimate, kept):
    from taxi_duration import tune

    seen = setup_run(
        monkeypatch,
        tmp_path,
        trials=[DEEP, GOOD],
        scores_for=lambda p: [3.2] * 3 if is_deep(p) else [3.0] * 3,
        estimate_for=lambda p: estimate if is_deep(p) else SMALL,
    )
    saved, loaded = wrap_saves(monkeypatch)

    result = tune.run_tuning()

    assert (DEEP in saved) is kept
    assert (DEEP in loaded) is kept
    assert result["name"] == "trial_1"  # the study went on and the good trial won
    first = read_trials(seen)["trials"][0]
    assert first["params"] == DEEP
    assert len(first["fold_rmse"]) == 3
    if kept:
        assert first["size_bytes"] == SMALL  # measured on the saved file
        assert first["predict_ms"] is not None
    else:
        assert first["size_bytes"] == estimate
        assert first["predict_ms"] is None
        assert tune.passes_limits(first["size_bytes"], first["predict_ms"]) is False
        assert len(seen.client.runs) == 2  # still logged, without a None metric


def test_the_log_never_receives_a_none_metric(monkeypatch, tmp_path):
    from taxi_duration import tune

    seen = setup_run(
        monkeypatch,
        tmp_path,
        trials=[DEEP, GOOD],
        estimate_for=lambda p: 3 * LIMIT if is_deep(p) else SMALL,
    )

    def log_metric(run_id, key, value, *args, **kwargs):
        if value is None:
            raise TypeError(f"metric {key} is None")

    monkeypatch.setattr(seen.client, "log_metric", log_metric)

    tune.run_tuning()


def test_passes_limits_and_choose_final_handle_a_missing_speed():
    from taxi_duration import tune

    assert tune.passes_limits(1_000, None) is False
    assert tune.passes_limits(LIMIT + 1, None) is False

    def rec(name, mean, ms):
        return {
            "name": name,
            "fold_rmse": [mean] * 3,
            "size_bytes": 1_000,
            "predict_ms": ms,
        }

    baseline = rec("baseline", 100.0, 1.0)
    unmeasured = rec("unmeasured", 50.0, None)
    ok = rec("ok", 90.0, 1.0)
    assert tune.choose_final(baseline, [unmeasured, ok])["name"] == "ok"
    assert tune.choose_final(baseline, [unmeasured])["name"] == "baseline"
    slow_base = rec("baseline", 100.0, 80.0)
    assert tune.choose_final(slow_base, [unmeasured]) is None


def disk_full(p):
    return is_deep(p)


def test_disk_error_in_a_trial_is_recorded_and_the_study_continues(
    monkeypatch, tmp_path
):
    from taxi_duration import artifacts, tune

    seen = setup_run(monkeypatch, tmp_path, trials=[DEEP, GOOD])
    real_save = artifacts.save_model

    def save(fitted, folder):
        if disk_full(fitted.params):
            raise RuntimeError("not enough disk space to save the model")
        return real_save(fitted, folder)

    monkeypatch.setattr(artifacts, "save_model", save)

    result = tune.run_tuning()

    assert result["name"] == "trial_1"
    trials = read_trials(seen)["trials"]
    assert [t["name"] for t in trials] == ["trial_0", "trial_1"]
    failed = trials[0]
    assert failed["disk_error"] is True
    assert failed["predict_ms"] is None
    assert tune.passes_limits(failed["size_bytes"], failed["predict_ms"]) is False
    assert len(failed["fold_rmse"]) == 3
    assert not trials[1].get("disk_error")
    assert len(seen.client.runs) == 2


def test_trials_file_is_written_after_a_disk_error(monkeypatch, tmp_path):
    from taxi_duration import artifacts, tune

    def crash_in_second(p):
        if p == GOOD:
            raise RuntimeError("boom in trial 2")
        return [3.0, 3.0, 3.0]

    seen = setup_run(
        monkeypatch, tmp_path, trials=[DEEP, GOOD], scores_for=crash_in_second
    )

    def save(fitted, folder):
        raise RuntimeError("not enough disk space to save the model")

    monkeypatch.setattr(artifacts, "save_model", save)

    with pytest.raises(RuntimeError, match="boom in trial 2"):
        tune.run_tuning()

    trials = read_trials(seen)["trials"]
    assert [t["name"] for t in trials] == ["trial_0"]
    assert trials[0]["disk_error"] is True
