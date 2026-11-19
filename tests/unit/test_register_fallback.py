"""register_from_results: saved-file check and the approved fallback (findings 5 and 8).

Interface tune.py must provide:
- register_from_results(tracking_uri=None, approved_fallback=False) -> dict.
- Saved-file check (final and fallback paths): model_dir/model.skops must exist and
  its size must equal the recorded size_bytes; otherwise ValueError (message contains
  "model.skops" when missing and "size" on a mismatch) before register_champion is
  called, so before the store is touched.
- "fallback" section and approved_fallback False: ValueError containing "approval".
- approved_fallback True with a fallback section: registers fallback["selected"]
  (model_dir, size_bytes, predict_ms) through register_champion(..., tracking_uri=...).
  ValueError containing "limits" when fallback["misses"] is true.
- approved_fallback True without a fallback section: ValueError containing
  "no fallback".
"""

import json

import pytest

REGISTERED = {
    "name": "taxi-duration",
    "alias": "champion",
    "version": "2",
    "sha256": "cd" * 32,
}


@pytest.fixture
def results(monkeypatch, tmp_path):
    from taxi_duration import config

    folder = tmp_path / "results"
    folder.mkdir()
    monkeypatch.setattr(config, "RESULTS_DIR", str(folder))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    return folder


@pytest.fixture
def registered(monkeypatch):
    from taxi_duration import tune

    calls = []

    def register_champion(model_dir, size_bytes, predict_ms, tracking_uri=None):
        calls.append((model_dir, size_bytes, predict_ms, tracking_uri))
        return REGISTERED

    monkeypatch.setattr(tune, "register_champion", register_champion)
    return calls


def model_folder(tmp_path, size=2_000, name="candidate"):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "model.skops").write_bytes(b"x" * size)
    return folder


def record(folder, size=2_000, ms=3.0):
    return {
        "params": {"max_depth": 12},
        "size_bytes": size,
        "predict_ms": ms,
        "model_dir": str(folder),
    }


def write(results, data):
    (results / "trials.json").write_text(json.dumps(data))


def write_final(results, **record_kwargs):
    write(
        results, {"chosen": "trial_1", "final": record(**record_kwargs), "trials": []}
    )


# Finding 5: the saved file is what trials.json says it is.


def test_missing_model_file_stops_before_the_store(results, registered, tmp_path):
    from taxi_duration import tune

    folder = tmp_path / "empty"
    folder.mkdir()
    write_final(results, folder=folder)

    with pytest.raises(ValueError, match="model.skops"):
        tune.register_from_results()

    assert registered == []


@pytest.mark.parametrize("recorded", [1_999, 2_001, 0])
def test_size_mismatch_stops_before_the_store(results, registered, tmp_path, recorded):
    from taxi_duration import tune

    folder = model_folder(tmp_path, size=2_000)
    write_final(results, folder=folder, size=recorded)

    with pytest.raises(ValueError, match="(?i)size"):
        tune.register_from_results()

    assert registered == []


def test_matching_file_registers(results, registered, tmp_path):
    from taxi_duration import tune

    folder = model_folder(tmp_path)
    write_final(results, folder=folder)

    assert tune.register_from_results() == REGISTERED
    assert registered == [(str(folder), 2_000, 3.0, None)]


# Finding 8: the approved fallback.


def fallback_data(folder, *, misses=False, size=2_000, ms=3.0):
    return {
        "chosen": None,
        "fallback": {
            "from": "baseline",
            "misses": misses,
            "attempts": [],
            "selected": record(folder, size=size, ms=ms),
        },
    }


def test_plain_register_still_refuses_the_fallback(results, registered, tmp_path):
    from taxi_duration import tune

    write(results, fallback_data(model_folder(tmp_path)))

    with pytest.raises(ValueError, match="approval"):
        tune.register_from_results(approved_fallback=False)

    assert registered == []


def test_approved_fallback_registers_the_selected_attempt(
    results, registered, tmp_path
):
    from taxi_duration import tune

    folder = model_folder(tmp_path)
    write(results, fallback_data(folder))

    got = tune.register_from_results(approved_fallback=True)

    assert got == REGISTERED
    assert registered == [(str(folder), 2_000, 3.0, None)]

    tune.register_from_results(
        tracking_uri="http://127.0.0.1:5001", approved_fallback=True
    )
    assert registered[-1][3] == "http://127.0.0.1:5001"


def test_approved_fallback_refuses_a_model_that_misses_the_limits(
    results, registered, tmp_path
):
    from taxi_duration import tune

    folder = model_folder(tmp_path)
    write(results, fallback_data(folder, misses=True, ms=80.0))

    with pytest.raises(ValueError, match="limits"):
        tune.register_from_results(approved_fallback=True)

    assert registered == []


def test_approved_fallback_checks_the_saved_file(results, registered, tmp_path):
    from taxi_duration import tune

    folder = model_folder(tmp_path, size=2_000)
    write(results, fallback_data(folder, size=1_000))

    with pytest.raises(ValueError, match="(?i)size"):
        tune.register_from_results(approved_fallback=True)

    assert registered == []


@pytest.mark.parametrize("with_final", [True, False])
def test_approved_fallback_needs_a_fallback_section(
    results, registered, tmp_path, with_final
):
    from taxi_duration import tune

    if with_final:
        write_final(results, folder=model_folder(tmp_path))
    else:
        write(results, {"chosen": None, "trials": []})

    with pytest.raises(ValueError, match="no fallback"):
        tune.register_from_results(approved_fallback=True)

    assert registered == []
