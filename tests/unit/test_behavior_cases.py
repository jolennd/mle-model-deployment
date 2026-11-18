"""The behavioral cases file parses and is read, not copied into code. No model needed."""

import tomllib
from pathlib import Path

BEHAVIORAL = Path(__file__).resolve().parents[1] / "behavioral"
APPROVED_IDS = {"mft_jfk_17_miles", "dir_longer_distance", "dir_rush_hour_not_faster"}
INPUT_FIELDS = {"pickup_zone_id", "dropoff_zone_id", "trip_distance", "pickup_at"}
REQUIRED = {
    "mft": {"id", "type", "input", "expected_min", "why"},
    "dir": {"id", "type", "input", "perturbed", "direction", "min_step", "why"},
}


def load_cases():
    return tomllib.loads((BEHAVIORAL / "cases.toml").read_text())["case"]


def test_cases_file_parses_with_the_approved_cases():
    cases = load_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    assert set(ids) == APPROVED_IDS
    for case in cases:
        assert case["type"] in REQUIRED, case["id"]
        assert REQUIRED[case["type"]] <= case.keys(), case["id"]
        assert set(case["input"]) == INPUT_FIELDS, case["id"]
        if case["type"] == "dir":
            assert set(case["perturbed"]) == INPUT_FIELDS, case["id"]
            assert case["direction"] in ("up", "down"), case["id"]


def test_behavior_tests_read_the_file_and_do_not_copy_it():
    source = (BEHAVIORAL / "test_behavior.py").read_text()
    assert "cases.toml" in source
    for case in load_cases():
        assert case["id"] not in source, f"{case['id']} is copied into code"
