"""Staging counts: raw rows equal kept rows plus dropped rows, per reason.

A synthetic raw_trips (fixed seed, planted row counts) is written to a temporary DuckDB
file and transformed by the project's own command (`transform/dbt_env.py run`) through
DB_CONN. The test reads stg_trips_drop_counts. No real file, no network.
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

WINDOW_START = datetime(2025, 1, 1)
WINDOW_END = datetime(2025, 2, 1)

# Planted counts, all different so a mix-up between reasons shows.
N_VALID = 50
N_EXACT_DUPLICATES = 3  # stay in, counted as kept
N_BEFORE_WINDOW = 4
N_AFTER_WINDOW = 3
N_DROPOFF_BEFORE_PICKUP = 2
N_OUTSIDE_AND_REVERSED = 1  # exclusive reasons: counted under the date window only

EXPECTED_RAW = (
    N_VALID
    + N_EXACT_DUPLICATES
    + N_BEFORE_WINDOW
    + N_AFTER_WINDOW
    + N_DROPOFF_BEFORE_PICKUP
    + N_OUTSIDE_AND_REVERSED
)
EXPECTED_DATE_WINDOW = N_BEFORE_WINDOW + N_AFTER_WINDOW + N_OUTSIDE_AND_REVERSED
EXPECTED_DROPOFF = N_DROPOFF_BEFORE_PICKUP
EXPECTED_KEPT = N_VALID + N_EXACT_DUPLICATES

TYPES = {
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
SOURCE_COLUMNS = [
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


def _row(rng, pickup, dropoff):
    fare = round(rng.uniform(5, 60), 2)
    return (
        rng.choice([1, 2, 6]),
        pickup,
        dropoff,
        rng.randint(1, 4),
        round(rng.uniform(0.3, 20), 2),
        rng.choice([1, 2, 5]),
        rng.choice(["N", "Y"]),
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


def synthetic_rows(seed=2025):
    rng = random.Random(seed)

    def inside():
        return WINDOW_START + timedelta(minutes=rng.randint(0, 30 * 24 * 60))

    rows = []
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


@pytest.fixture(scope="module")
def built_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("staging_counts") / "counts.duckdb"
    con = duckdb.connect(str(path))
    try:
        cols = ", ".join(f'"{c}" {TYPES.get(c, "DOUBLE")}' for c in SOURCE_COLUMNS)
        con.execute(
            f"CREATE TABLE raw_trips ({cols}, loaded_at TIMESTAMP, source_partition VARCHAR)"
        )
        marks = ", ".join(["?"] * (len(SOURCE_COLUMNS) + 2))
        loaded_at = datetime(2025, 3, 1, 12, 0, 0)
        con.executemany(
            f"INSERT INTO raw_trips VALUES ({marks})",
            [(*r, loaded_at, "2025-01") for r in synthetic_rows()],
        )
    finally:
        con.close()

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


def test_raw_rows_equal_kept_plus_dropped_per_reason(built_db):
    con = duckdb.connect(str(built_db), read_only=True)
    try:
        counts = dict(
            con.execute(
                "SELECT reason, row_count FROM stg_trips_drop_counts"
            ).fetchall()
        )
        raw_table = con.execute("SELECT count(*) FROM raw_trips").fetchone()[0]
        stg_rows = con.execute("SELECT count(*) FROM stg_trips").fetchone()[0]
    finally:
        con.close()

    assert set(counts) == {
        "raw_rows",
        "dropped_date_window",
        "dropped_dropoff_before_pickup",
        "kept",
    }
    # The fixture is not vacuous: every reason has rows and the numbers differ.
    planted = [EXPECTED_DATE_WINDOW, EXPECTED_DROPOFF, EXPECTED_KEPT]
    assert all(n > 0 for n in planted) and len(set(planted)) == 3
    assert raw_table == EXPECTED_RAW

    assert counts["raw_rows"] == EXPECTED_RAW
    assert counts["dropped_date_window"] == EXPECTED_DATE_WINDOW
    assert counts["dropped_dropoff_before_pickup"] == EXPECTED_DROPOFF
    assert counts["kept"] == EXPECTED_KEPT
    assert counts["raw_rows"] == (
        counts["kept"]
        + counts["dropped_date_window"]
        + counts["dropped_dropoff_before_pickup"]
    ), "rows went missing: raw is not kept plus every dropped reason"
    assert stg_rows == counts["kept"], "stg_trips row count differs from kept"
