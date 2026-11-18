"""Leakage gates on the split module: the lock guards the test week, the embargo covers the horizon."""

import duckdb
import pandas as pd
import pytest

from taxi_duration import config


def write_mart(path, trips):
    """Create mart_trips in the temporary database. trips: (id, pickup, dropoff)."""
    con = duckdb.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE mart_trips (trip_id VARCHAR, pickup_at TIMESTAMP, dropoff_at TIMESTAMP, "
            "pickup_zone_id INTEGER, dropoff_zone_id INTEGER, trip_distance DOUBLE)"
        )
        for trip_id, pickup, dropoff in trips:
            con.execute(
                "INSERT INTO mart_trips VALUES (?, ?, ?, 100, 100, 2.0)",
                [trip_id, pickup, dropoff],
            )
    finally:
        con.close()


def test_load_test_refuses_before_lock(db_conn, tmp_path, monkeypatch):
    from taxi_duration.splits import load_test

    monkeypatch.setattr(config, "LOCK_PATH", str(tmp_path / "baseline_lock.json"))
    write_mart(db_conn, [("w1", "2025-01-26 10:00:00", "2025-01-26 10:10:00")])

    with pytest.raises(RuntimeError, match="baseline lock"):
        load_test()

    # the refusal comes before any database access
    monkeypatch.delenv("DB_CONN")
    with pytest.raises(RuntimeError, match="baseline lock"):
        load_test()


def test_embargo_covers_label_horizon(db_conn):
    assert config.EMBARGO_MINUTES >= config.LABEL_HORIZON_MINUTES

    from taxi_duration.splits import load_train

    write_mart(
        db_conn,
        [
            ("ordinary", "2025-01-10 08:00:00", "2025-01-10 08:10:00"),
            ("straddle", "2025-01-24 23:50:00", "2025-01-25 00:20:00"),
            ("test_week", "2025-01-26 10:00:00", "2025-01-26 10:10:00"),
        ],
    )
    out = load_train()

    assert set(out["trip_id"]) == {"ordinary"}
    assert "duration_minutes" in out.columns
    assert (out["dropoff_at"] < pd.Timestamp(config.TEST_START)).all()
