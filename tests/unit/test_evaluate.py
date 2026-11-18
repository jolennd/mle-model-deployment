"""Reported metrics and the single test read (docs/tuning-plan.md, batch 2).

Interface evaluate.py must provide:
- report_metrics(actual, predicted, pickup_zone_id) -> dict with keys
  "rmse", "mae", "r2", "median_ae", "mean_error" (predicted minus actual),
  "rmse_by_segment" and "rows_by_segment" (both keyed by SEGMENTS below), "rows".
  Segments use the actual duration; airports are pickup zones 1, 132 and 138.
  An empty segment has rmse None and rows 0. No MAPE key anywhere.
- evaluate_test(model_loader, baseline_loader, *, load_test, results_dir) -> dict.
  Calls load_test() once, model_loader() and baseline_loader() once each, builds
  features.build_features(frame) once, scores both with .predict(features) against
  frame["duration_minutes"], writes <results_dir>/metrics.json and returns the same
  dict: {"chosen": report, "baseline": report, "rows_used": n, "test_reads": 1}.
  Raises FileExistsError before calling load_test when metrics.json exists.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SEGMENTS = [
    "duration_1_10",
    "duration_10_30",
    "duration_30_60",
    "airport",
    "other",
]

ACTUAL = np.array([1.0, 5.0, 9.99, 10.0, 20.0, 29.99, 30.0, 45.0, 60.0])
ERROR = np.array([2.0, -1.0, 3.0, 4.0, -2.0, 1.0, -5.0, 6.0, -3.0])
PREDICTED = ACTUAL + ERROR
ZONES = np.array([132, 5, 1, 7, 138, 9, 5, 2, 132])
LOW, MID, HIGH = [0, 1, 2], [3, 4, 5], [6, 7, 8]
AIRPORT = [0, 2, 4, 8]
OTHER = [1, 3, 5, 6, 7]


def rmse(idx):
    return float(np.sqrt(np.mean(ERROR[idx] ** 2)))


def test_reported_metrics_include_segments():
    from taxi_duration import evaluate

    got = evaluate.report_metrics(ACTUAL, PREDICTED, ZONES)

    assert got["rmse"] == pytest.approx(rmse(slice(None)))
    assert got["mae"] == pytest.approx(np.mean(np.abs(ERROR)))
    ss_res = np.sum(ERROR**2)
    ss_tot = np.sum((ACTUAL - ACTUAL.mean()) ** 2)
    assert got["r2"] == pytest.approx(1 - ss_res / ss_tot)
    assert got["median_ae"] == pytest.approx(np.median(np.abs(ERROR)))
    assert got["mean_error"] == pytest.approx(np.mean(ERROR))  # prediction - actual
    assert got["mean_error"] > 0
    assert got["rows"] == 9

    seg, rows = got["rmse_by_segment"], got["rows_by_segment"]
    assert set(seg) == set(SEGMENTS) == set(rows)
    # 10 and 30 go to the upper segment, 60 is included
    assert rows == {
        "duration_1_10": 3,
        "duration_10_30": 3,
        "duration_30_60": 3,
        "airport": 4,
        "other": 5,
    }
    assert seg["duration_1_10"] == pytest.approx(rmse(LOW))
    assert seg["duration_10_30"] == pytest.approx(rmse(MID))
    assert seg["duration_30_60"] == pytest.approx(rmse(HIGH))
    assert seg["airport"] == pytest.approx(rmse(AIRPORT))
    assert seg["other"] == pytest.approx(rmse(OTHER))

    def keys(value):
        if isinstance(value, dict):
            for k, v in value.items():
                yield str(k).lower()
                yield from keys(v)

    assert not any("mape" in k for k in keys(got))


def test_empty_segment_is_explicit():
    from taxi_duration import evaluate

    actual = np.array([12.0, 20.0, 25.0])
    got = evaluate.report_metrics(actual, actual + 1.0, np.array([5, 6, 7]))

    assert got["rmse_by_segment"]["duration_1_10"] is None
    assert got["rows_by_segment"]["duration_1_10"] == 0
    assert got["rmse_by_segment"]["duration_30_60"] is None
    assert got["rows_by_segment"]["airport"] == 0
    assert got["rmse_by_segment"]["airport"] is None
    assert got["rmse_by_segment"]["duration_10_30"] == pytest.approx(1.0)
    assert got["rmse_by_segment"]["other"] == pytest.approx(1.0)


class ConstantModel:
    """Records the frame it was asked to predict."""

    def __init__(self, value):
        self.value = value
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame)
        return np.full(len(frame), self.value)


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


def test_test_rows_read_once(tmp_path):
    from taxi_duration import evaluate

    frame = make_test_frame()
    reads = []

    def load_test():
        reads.append(1)
        return frame

    chosen, baseline = ConstantModel(20.0), ConstantModel(10.0)
    loads = []

    def model_loader():
        loads.append("chosen")
        return chosen

    def baseline_loader():
        loads.append("baseline")
        return baseline

    results = tmp_path / "results"
    got = evaluate.evaluate_test(
        model_loader, baseline_loader, load_test=load_test, results_dir=results
    )

    assert len(reads) == 1
    assert sorted(loads) == ["baseline", "chosen"]
    assert len(chosen.frames) == len(baseline.frames) == 1
    assert len(chosen.frames[0]) == 6
    pd.testing.assert_frame_equal(chosen.frames[0], baseline.frames[0])

    actual = frame["duration_minutes"].to_numpy()
    assert got["chosen"]["rmse"] == pytest.approx(np.sqrt(np.mean((20 - actual) ** 2)))
    assert got["baseline"]["rmse"] == pytest.approx(
        np.sqrt(np.mean((10 - actual) ** 2))
    )
    assert got["chosen"]["rows_by_segment"]["airport"] == 2

    saved = json.loads((results / "metrics.json").read_text())
    assert saved["test_reads"] == 1
    assert saved["rows_used"] == 6
    assert saved["chosen"]["rmse"] == pytest.approx(got["chosen"]["rmse"])
    assert saved["baseline"]["rmse"] == pytest.approx(got["baseline"]["rmse"])

    # A second call refuses and never touches the test rows again.
    with pytest.raises(FileExistsError):
        evaluate.evaluate_test(
            model_loader, baseline_loader, load_test=load_test, results_dir=results
        )
    assert len(reads) == 1


def test_existing_metrics_file_blocks_the_read(tmp_path):
    from taxi_duration import evaluate

    (tmp_path / "metrics.json").write_text(json.dumps({"test_reads": 1}))
    reads = []

    with pytest.raises(FileExistsError):
        evaluate.evaluate_test(
            lambda: ConstantModel(1.0),
            lambda: ConstantModel(1.0),
            load_test=lambda: reads.append(1),
            results_dir=tmp_path,
        )
    assert reads == []


def test_tuning_and_training_code_never_read_the_test_rows():
    root = Path(__file__).resolve().parents[2] / "src" / "taxi_duration"
    for name in ("tune.py", "train.py"):
        source = (root / name).read_text()
        assert "load_test" not in source, f"{name} mentions load_test"
