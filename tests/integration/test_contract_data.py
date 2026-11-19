"""Contract of the data layer: raw_trips, stg_trips and mart_trips.

The tables are built from a small synthetic raw table with a fixed seed. The test writes
raw_trips into a temporary DuckDB file, then runs the project's own transform command
(`transform/dbt_env.py run`) against it through DB_CONN. No real file, no network.
"""

import random
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]

RAW_SOURCE_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
]
RAW_COLUMNS = RAW_SOURCE_COLUMNS + ["loaded_at", "source_partition"]

STG_COLUMNS = [
    "vendor_id",
    "pickup_at",
    "dropoff_at",
    "passenger_count",
    "trip_distance",
    "rate_code_id",
    "store_and_fwd_flag",
    "pickup_zone_id",
    "dropoff_zone_id",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
]

MART_COLUMNS = [
    "trip_id",
    "pickup_at",
    "dropoff_at",
    "pickup_zone_id",
    "dropoff_zone_id",
    "trip_distance",
]

FORBIDDEN_IN_MART = {
    "fare_amount",
    "tip_amount",
    "tolls_amount",
    "total_amount",
    "payment_type",
    "ratecodeid",
    "rate_code_id",
    "extra",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
    "duration",
    "duration_min",
    "duration_minutes",
    "trip_duration",
    "label",
}

WINDOW_START = datetime(2025, 1, 1)
WINDOW_END = datetime(2025, 2, 1)

N_VALID = 120
N_EXACT_DUPLICATES = 6
N_BEFORE_WINDOW = 7
N_AFTER_WINDOW = 5
N_DROPOFF_BEFORE_PICKUP = 4
N_OUTSIDE_AND_REVERSED = 2  # counted under the date window only

# Rows the staging step keeps: valid trips plus their exact duplicates.
EXPECTED_KEPT = N_VALID + N_EXACT_DUPLICATES
ROW_FLOOR = EXPECTED_KEPT


def _row(rng, pickup, dropoff):
    fare = round(rng.uniform(5, 60), 2)
    return (
        rng.choice([1, 2, 6]),
        pickup,
        dropoff,
        rng.randint(1, 4),
        round(rng.uniform(0.3, 20), 2),
        rng.choice([1, 1, 1, 2, 5]),
        rng.choice(["N", "N", "Y"]),
        rng.randint(1, 263),
        rng.randint(1, 263),
        rng.choice([1, 2]),
        fare,
        rng.choice([0.0, 1.0, 2.5]),
        0.5,
        round(rng.uniform(0, 10), 2),
        0.0,
        1.0,
        round(fare + 3, 2),
        2.5,
        0.0,
        0.0,
    )


def synthetic_raw_rows(seed=42):
    rng = random.Random(seed)
    rows = []

    def inside():
        return WINDOW_START + timedelta(minutes=rng.randint(0, 31 * 24 * 60 - 120))

    for _ in range(N_VALID):
        pickup = inside()
        rows.append(_row(rng, pickup, pickup + timedelta(minutes=rng.randint(2, 50))))
    rows += rows[:N_EXACT_DUPLICATES]
    for _ in range(N_BEFORE_WINDOW):
        pickup = WINDOW_START - timedelta(minutes=rng.randint(60, 5000))
        rows.append(_row(rng, pickup, pickup + timedelta(minutes=10)))
    for _ in range(N_AFTER_WINDOW):
        pickup = WINDOW_END + timedelta(minutes=rng.randint(0, 5000))
        rows.append(_row(rng, pickup, pickup + timedelta(minutes=10)))
    for _ in range(N_DROPOFF_BEFORE_PICKUP):
        pickup = inside()
        rows.append(_row(rng, pickup, pickup - timedelta(minutes=rng.randint(1, 30))))
    for _ in range(N_OUTSIDE_AND_REVERSED):
        pickup = WINDOW_START - timedelta(minutes=500)
        rows.append(_row(rng, pickup, pickup - timedelta(minutes=5)))
    return rows


def _write_raw(path, rows):
    con = duckdb.connect(str(path))
    try:
        types = {
            "VendorID": "INTEGER",
            "tpep_pickup_datetime": "TIMESTAMP",
            "tpep_dropoff_datetime": "TIMESTAMP",
            "passenger_count": "BIGINT",
            "trip_distance": "DOUBLE",
            "RatecodeID": "BIGINT",
            "store_and_fwd_flag": "VARCHAR",
            "PULocationID": "INTEGER",
            "DOLocationID": "INTEGER",
            "payment_type": "BIGINT",
        }
        cols = ", ".join(f'"{c}" {types.get(c, "DOUBLE")}' for c in RAW_SOURCE_COLUMNS)
        con.execute(
            f"CREATE TABLE raw_trips ({cols}, loaded_at TIMESTAMP, source_partition VARCHAR)"
        )
        marks = ", ".join(["?"] * (len(RAW_SOURCE_COLUMNS) + 2))
        loaded_at = datetime(2025, 3, 1, 12, 0, 0)
        con.executemany(
            f"INSERT INTO raw_trips VALUES ({marks})",
            [(*r, loaded_at, "2025-01") for r in rows],
        )
    finally:
        con.close()


@pytest.fixture(scope="module")
def built_db(tmp_path_factory):
    """A database file with a synthetic raw_trips, transformed by the project's own command."""
    path = tmp_path_factory.mktemp("warehouse") / "contract.duckdb"
    _write_raw(path, synthetic_raw_rows())
    mp = pytest.MonkeyPatch()
    mp.setenv("DB_CONN", f"duckdb:///{path}")
    try:
        result = subprocess.run(
            [sys.executable, "transform/dbt_env.py", "run"],
            check=False,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        mp.undo()
    assert result.returncode == 0, (
        f"transform run failed:\n{result.stdout}\n{result.stderr}"
    )
    return path


def _query(path, sql):
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _columns(path, table):
    rows = _query(
        path,
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_name = '{table}' ORDER BY ordinal_position",
    )
    return [r[0] for r in rows]


def test_tables_exist_with_columns(built_db):
    assert _columns(built_db, "raw_trips") == RAW_COLUMNS
    assert _columns(built_db, "stg_trips") == STG_COLUMNS
    assert _columns(built_db, "mart_trips") == MART_COLUMNS


def test_key_unique(built_db):
    total, distinct, nulls = _query(
        built_db,
        "SELECT count(*), count(DISTINCT trip_id), count(*) - count(trip_id) FROM mart_trips",
    )[0]
    assert nulls == 0
    assert total == distinct, f"{total - distinct} duplicate trip_id values"
    # exact duplicate source rows stay in, each with its own key
    assert total == EXPECTED_KEPT


def test_event_time_and_loaded_at_present(built_db):
    assert (
        _query(built_db, "SELECT count(*) FROM raw_trips WHERE loaded_at IS NULL")[0][0]
        == 0
    )
    assert (
        _query(
            built_db, "SELECT count(*) FROM raw_trips WHERE source_partition IS NULL"
        )[0][0]
        == 0
    )
    null_times = _query(
        built_db,
        "SELECT count(*) FROM mart_trips WHERE pickup_at IS NULL OR dropoff_at IS NULL",
    )[0][0]
    assert null_times == 0
    low, high = _query(
        built_db, "SELECT min(pickup_at), max(pickup_at) FROM mart_trips"
    )[0]
    assert WINDOW_START <= low and high < WINDOW_END


def test_row_count_above_floor(built_db):
    for table in ("stg_trips", "mart_trips"):
        count = _query(built_db, f"SELECT count(*) FROM {table}")[0][0]
        assert count >= ROW_FLOOR, f"{table} has {count} rows, floor is {ROW_FLOOR}"
    assert (
        _query(built_db, "SELECT count(*) FROM stg_trips")[0][0]
        == _query(built_db, "SELECT count(*) FROM mart_trips")[0][0]
    ), "rows lost between staging and mart"


def test_mart_has_no_label_columns(built_db):
    names = {c.lower() for c in _columns(built_db, "mart_trips")}
    assert names, "mart_trips does not exist"
    assert names.isdisjoint(FORBIDDEN_IN_MART), sorted(names & FORBIDDEN_IN_MART)
