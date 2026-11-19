"""Test-read marker (ADR 8 point 18).

Interface evaluate.evaluate_test must provide (loader order stays: model, baseline,
then the marker, then load_test):
- Before it calls load_test it writes <results_dir>/test_read_started.json (creating
  the folder): a JSON object with "started_at", an ISO 8601 timestamp.
- Marker present and metrics.json absent -> RuntimeError whose message contains
  "test_read_started" and "ADR 8". Nothing is loaded and load_test is not called.
- Both present -> FileExistsError (the metrics guard, as before).
- A loader that fails leaves no marker. A failure after load_test (for example in
  predict) leaves the marker and writes no metrics.json, so the next call refuses.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

MARKER = "test_read_started.json"


def make_frame():
    pickup = pd.date_range("2025-01-25 08:00", periods=4, freq="h")
    return pd.DataFrame(
        {
            "trip_id": [f"t{i}" for i in range(4)],
            "pickup_at": pickup,
            "dropoff_at": pickup + pd.Timedelta(minutes=15),
            "pickup_zone_id": [132, 10, 20, 138],
            "dropoff_zone_id": [30, 30, 40, 40],
            "trip_distance": [10.0, 1.0, 2.0, 3.0],
            "duration_minutes": [30.0, 5.0, 10.0, 15.0],
        }
    )


class Constant:
    def __init__(self, value, calls=None):
        self.value = value
        self.calls = calls

    def predict(self, frame):
        return np.full(len(frame), self.value)


class Broken:
    def predict(self, frame):
        raise RuntimeError("predict failed")


def call(results, *, load_test, model=None, baseline=None, loads=None):
    from taxi_duration import evaluate

    loads = loads if loads is not None else []

    def model_loader():
        loads.append("model")
        return model or Constant(20.0)

    def baseline_loader():
        loads.append("baseline")
        return baseline or Constant(10.0)

    return evaluate.evaluate_test(
        model_loader, baseline_loader, load_test=load_test, results_dir=results
    )


def test_marker_exists_when_the_test_rows_are_read(tmp_path):
    results = tmp_path / "results"  # does not exist yet
    seen = {}

    def load_test():
        marker = results / MARKER
        seen["exists"] = marker.exists()
        seen["content"] = json.loads(marker.read_text()) if marker.exists() else None
        return make_frame()

    call(results, load_test=load_test)

    assert seen["exists"] is True
    assert isinstance(seen["content"], dict)
    datetime.fromisoformat(seen["content"]["started_at"])
    assert (results / "metrics.json").exists()  # the marker stays after success


def test_marker_without_metrics_refuses_and_never_reads(tmp_path):
    (tmp_path / MARKER).write_text(json.dumps({"started_at": "2026-11-18T10:00:00"}))
    reads, loads = [], []

    with pytest.raises(RuntimeError) as caught:
        call(tmp_path, load_test=lambda: reads.append(1), loads=loads)

    assert "test_read_started" in str(caught.value)
    assert "ADR 8" in str(caught.value)
    assert reads == []
    assert loads == []
    assert not (tmp_path / "metrics.json").exists()


def test_marker_and_metrics_together_hit_the_metrics_guard(tmp_path):
    (tmp_path / MARKER).write_text("{}")
    (tmp_path / "metrics.json").write_text(json.dumps({"test_reads": 1}))
    reads = []

    with pytest.raises(FileExistsError):
        call(tmp_path, load_test=lambda: reads.append(1))

    assert reads == []


def test_a_failing_predict_leaves_the_marker_and_the_next_call_refuses(tmp_path):
    reads = []

    def load_test():
        reads.append(1)
        return make_frame()

    with pytest.raises(RuntimeError, match="predict failed"):
        call(tmp_path, load_test=load_test, model=Broken())

    assert (tmp_path / MARKER).exists()
    assert not (tmp_path / "metrics.json").exists()
    assert reads == [1]

    loads = []
    with pytest.raises(RuntimeError, match="test_read_started"):
        call(tmp_path, load_test=load_test, loads=loads)
    assert reads == [1]
    assert loads == []


def test_a_failing_loader_leaves_no_marker(tmp_path):
    from taxi_duration import evaluate

    def bad_loader():
        raise RuntimeError("could not be loaded")

    with pytest.raises(RuntimeError, match="could not be loaded"):
        evaluate.evaluate_test(
            bad_loader,
            lambda: Constant(1.0),
            load_test=lambda: make_frame(),
            results_dir=tmp_path,
        )

    assert not (tmp_path / MARKER).exists()
