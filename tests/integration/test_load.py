"""Load step: one partition into raw_trips, repeatable, and loud on schema drift.

Source files are tiny synthetic parquet files with a fixed seed, loaded into a temporary
DuckDB file named by DB_CONN (the db_conn fixture). No real data, no network.
"""

import importlib
import random
from datetime import datetime, timedelta

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taxi_duration import config

pytestmark = pytest.mark.integration

PARTITION = "2025-01"
OTHER_PARTITION = "2024-12"
N_ROWS = 12


def _load_module():
    return importlib.import_module("taxi_duration.ingest.load")


def _columns(rows=N_ROWS, seed=7):
    rng = random.Random(seed)
    start = datetime(2025, 1, 1)
    pickups = [start + timedelta(minutes=rng.randint(0, 40000)) for _ in range(rows)]
    return {
        "VendorID": pa.array([rng.choice([1, 2]) for _ in range(rows)], pa.int32()),
        "tpep_pickup_datetime": pa.array(pickups, pa.timestamp("us")),
        "tpep_dropoff_datetime": pa.array(
            [p + timedelta(minutes=rng.randint(2, 50)) for p in pickups],
            pa.timestamp("us"),
        ),
        "trip_distance": pa.array(
            [round(rng.uniform(0.3, 20), 2) for _ in range(rows)], pa.float64()
        ),
        "PULocationID": pa.array(
            [rng.randint(1, 263) for _ in range(rows)], pa.int32()
        ),
        "payment_type": pa.array([rng.choice([1, 2]) for _ in range(rows)], pa.int64()),
    }


def _write(path, columns):
    pq.write_table(pa.table(columns), path)
    return path


@pytest.fixture
def source(tmp_path):
    return _write(tmp_path / "yellow_tripdata_2025-01.parquet", _columns())


def _snapshot(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        types = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'raw_trips' ORDER BY ordinal_position"
        ).fetchall()
        rows = con.execute("SELECT * FROM raw_trips ORDER BY ALL").fetchall()
        return types, rows
    finally:
        con.close()


def test_load_is_idempotent(db_conn, source):
    load = _load_module()
    assert PARTITION in config.PARTITIONS

    assert load.load_partition(PARTITION, source) == N_ROWS
    first_types, first_rows = _snapshot(db_conn)
    column_names = [name for name, _ in first_types]
    assert "loaded_at" in column_names and "source_partition" in column_names

    # Rows of another partition, planted directly, must survive reloads.
    con = duckdb.connect(str(db_conn))
    try:
        con.execute(
            "INSERT INTO raw_trips "
            f"SELECT * REPLACE ('{OTHER_PARTITION}' AS source_partition) "
            f"FROM raw_trips WHERE source_partition = '{PARTITION}' LIMIT 3"
        )
    finally:
        con.close()
    _, before = _snapshot(db_conn)
    other_before = [
        r
        for r in before
        if r[column_names.index("source_partition")] == OTHER_PARTITION
    ]
    assert len(other_before) == 3

    assert load.load_partition(PARTITION, source) == N_ROWS
    types, after = _snapshot(db_conn)

    assert types == first_types
    idx_loaded = column_names.index("loaded_at")
    idx_part = column_names.index("source_partition")
    ours = [r for r in after if r[idx_part] == PARTITION]
    other_after = [r for r in after if r[idx_part] == OTHER_PARTITION]
    assert len(ours) == N_ROWS, "reload changed the row count"
    assert len(after) == N_ROWS + 3
    assert other_after == other_before, "another partition's rows were touched"

    def without_loaded_at(rows):
        return sorted(r[:idx_loaded] + r[idx_loaded + 1 :] for r in rows)

    first_ours = [r for r in first_rows if r[idx_part] == PARTITION]
    assert without_loaded_at(ours) == without_loaded_at(first_ours)
    assert all(r[idx_loaded] is not None for r in ours)


def _added(columns):
    return {**columns, "surprise_column": pa.array([1] * N_ROWS, pa.int64())}


def _removed(columns):
    return {k: v for k, v in columns.items() if k != "payment_type"}


def _retyped_to_text(columns):
    return {**columns, "trip_distance": pa.array(["1.5"] * N_ROWS, pa.string())}


def _retyped_int_width(columns):
    return {**columns, "VendorID": pa.array([1] * N_ROWS, pa.int64())}


@pytest.mark.parametrize(
    "change",
    [_added, _removed, _retyped_to_text, _retyped_int_width],
    ids=["column_added", "column_removed", "type_to_text", "int_width_changed"],
)
def test_schema_drift_fails(db_conn, source, tmp_path, change):
    load = _load_module()
    load.load_partition(PARTITION, source)
    types_before, rows_before = _snapshot(db_conn)

    drifted = _write(tmp_path / "drifted.parquet", change(_columns(seed=99)))
    with pytest.raises(load.SchemaDriftError):
        load.load_partition(PARTITION, drifted)

    assert issubclass(load.SchemaDriftError, ValueError)
    types_after, rows_after = _snapshot(db_conn)
    assert types_after == types_before, "table schema changed after a rejected load"
    assert rows_after == rows_before, "table rows changed after a rejected load"
