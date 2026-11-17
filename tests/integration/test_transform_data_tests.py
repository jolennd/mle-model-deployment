"""Each singular dbt data test must pass on clean data and fail when bad rows are planted.

A synthetic raw_trips (fixed seed, as in test_contract_data.py) is built once with the
project's own command (`transform/dbt_env.py run`). Each case copies that database file,
runs one named singular test (`transform/dbt_env.py test --select <name>`), then plants bad
rows with DuckDB and runs the same test again. Models are views, so a planted check target
is first turned into a table with the same content (in the copy only, no model file changes).
dbt exits 0 when a selector matches nothing, so a pass must also report PASS=1.
No real file, no network.
"""

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pytest
from test_contract_data import _write_raw, synthetic_raw_rows

pytestmark = [pytest.mark.integration, pytest.mark.transform]

ROOT = Path(__file__).resolve().parents[2]


def _dbt(path, *args):
    env = dict(os.environ, DB_CONN=f"duckdb:///{path}")
    return subprocess.run(
        [sys.executable, "transform/dbt_env.py", *args],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


@pytest.fixture(scope="module")
def clean_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("data_tests") / "clean.duckdb"
    _write_raw(path, synthetic_raw_rows())
    result = _dbt(path, "run")
    assert result.returncode == 0, (
        f"transform run failed:\n{result.stdout}\n{result.stderr}"
    )
    return path


@pytest.fixture
def db(clean_db, tmp_path):
    path = tmp_path / "work.duckdb"
    shutil.copy(clean_db, path)
    return path


def assert_dbt_test_passes(path, name):
    result = _dbt(path, "test", "--select", name)
    out = result.stdout + result.stderr
    assert result.returncode == 0 and re.search(r"PASS=1\b", out), (
        f"{name} did not run and pass on clean data:\n{out[-1500:]}"
    )


def assert_dbt_test_fails(path, name):
    result = _dbt(path, "test", "--select", name)
    out = result.stdout + result.stderr
    assert result.returncode != 0 and re.search(r"FAIL=1\b", out), (
        f"{name} did not fail on planted rows:\n{out[-1500:]}"
    )


def _execute(path, *statements):
    con = duckdb.connect(str(path))
    try:
        for sql in statements:
            con.execute(sql)
    finally:
        con.close()


def _freeze(*names):
    """SQL that turns views into tables with the same rows, so rows can be planted."""
    out = []
    for n in names:
        out += [
            f"CREATE TABLE {n}_frozen AS SELECT * FROM {n}",
            f"DROP VIEW {n}",
            f"ALTER TABLE {n}_frozen RENAME TO {n}",
        ]
    return out


def _copy_row(table, **overrides):
    """SQL that inserts a copy of one existing row with some columns changed."""
    sets = ", ".join(f"{v} AS {k}" for k, v in overrides.items())
    return f"INSERT INTO {table} SELECT * REPLACE ({sets}) FROM {table} LIMIT 1"


def test_pickup_in_window_fails_on_pickup_outside_window(db):
    name = "stg_pickup_in_window"
    assert_dbt_test_passes(db, name)
    february = "TIMESTAMP '2025-02-01 00:00:00'"
    december = "TIMESTAMP '2024-12-31 23:59:00'"
    _execute(
        db,
        *_freeze("stg_trips", "mart_trips"),
        _copy_row(
            "stg_trips",
            pickup_at=february,
            dropoff_at="TIMESTAMP '2025-02-01 00:10:00'",
        ),
        _copy_row(
            "mart_trips",
            pickup_at=december,
            dropoff_at="TIMESTAMP '2025-01-01 00:10:00'",
        ),
    )
    assert_dbt_test_fails(db, name)


def test_dropoff_not_before_pickup_fails_on_reversed_trip(db):
    name = "stg_dropoff_not_before_pickup"
    assert_dbt_test_passes(db, name)
    _execute(
        db,
        *_freeze("stg_trips", "mart_trips"),
        _copy_row(
            "stg_trips",
            pickup_at="TIMESTAMP '2025-01-10 12:00:00'",
            dropoff_at="TIMESTAMP '2025-01-10 11:55:00'",
        ),
        _copy_row(
            "mart_trips",
            pickup_at="TIMESTAMP '2025-01-10 12:00:00'",
            dropoff_at="TIMESTAMP '2025-01-10 11:55:00'",
        ),
    )
    assert_dbt_test_fails(db, name)


def test_zone_ids_in_range_fails_on_bad_zone(db):
    name = "stg_zone_ids_in_range"
    assert_dbt_test_passes(db, name)
    _execute(
        db,
        *_freeze("stg_trips"),
        _copy_row("stg_trips", pickup_zone_id="0"),
        _copy_row("stg_trips", dropoff_zone_id="300"),
        _copy_row("stg_trips", pickup_zone_id="NULL"),
    )
    assert_dbt_test_fails(db, name)


def test_drop_counts_reconcile_fails_on_stale_counts(db):
    name = "stg_drop_counts_reconcile"
    assert_dbt_test_passes(db, name)
    # Freeze the counts, then add a raw row the counts never saw.
    _execute(
        db,
        *_freeze("stg_trips_drop_counts"),
        _copy_row("raw_trips", loaded_at=f"TIMESTAMP '{datetime(2025, 3, 2)}'"),
    )
    assert_dbt_test_fails(db, name)


def test_counts_equal_kept_fails_when_a_row_is_missing(db, clean_db, tmp_path):
    name = "stg_counts_equal_kept"
    assert_dbt_test_passes(db, name)
    delete = "DELETE FROM {t} WHERE rowid IN (SELECT rowid FROM {t} LIMIT 1)"
    # A row missing from mart_trips only, then (fresh copy) from stg_trips only.
    _execute(db, *_freeze("mart_trips"), delete.format(t="mart_trips"))
    assert_dbt_test_fails(db, name)
    other = tmp_path / "other.duckdb"
    shutil.copy(clean_db, other)
    _execute(other, *_freeze("stg_trips"), delete.format(t="stg_trips"))
    assert_dbt_test_fails(other, name)
