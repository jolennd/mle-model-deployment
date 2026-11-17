"""Load one partition of a staged parquet file into raw_trips, repeatably."""

import argparse
import logging
from pathlib import Path

import duckdb

from .. import config, db
from .source_hash import verify_source_hash

log = logging.getLogger(__name__)

TABLE = "raw_trips"
ADDED_COLUMNS = ("loaded_at", "source_partition")


class SchemaDriftError(ValueError):
    """The source file columns or types differ from the existing raw table."""


def _source_schema(con, source_path: str) -> dict[str, str]:
    rows = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [source_path]
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def _table_schema(con) -> dict[str, str] | None:
    rows = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position",
        [TABLE],
    ).fetchall()
    if not rows:
        return None
    return {n: t for n, t in rows if n not in ADDED_COLUMNS}


def _check_schema(expected: dict[str, str], found: dict[str, str]) -> None:
    added = sorted(set(found) - set(expected))
    removed = sorted(set(expected) - set(found))
    changed = sorted(
        f"{n}: {expected[n]} -> {found[n]}"
        for n in set(expected) & set(found)
        if expected[n] != found[n]
    )
    if added or removed or changed:
        raise SchemaDriftError(
            f"schema drift in {TABLE}: added {added}, removed {removed}, "
            f"type changed {changed}"
        )


def _load(con, partition: str, source_path: str) -> int:
    rows_in = con.execute(
        "SELECT count(*) FROM read_parquet(?)", [source_path]
    ).fetchone()[0]
    existing = _table_schema(con)
    if existing is not None:
        _check_schema(existing, _source_schema(con, source_path))
    stamp = "CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP)"
    select = (
        f"SELECT *, {stamp} AS loaded_at, ? AS source_partition FROM read_parquet(?)"
    )
    con.execute("BEGIN")
    try:
        if existing is None:
            con.execute(
                f"CREATE TABLE {TABLE} AS {select} WHERE false",
                [partition, source_path],
            )
        con.execute(f"DELETE FROM {TABLE} WHERE source_partition = ?", [partition])
        con.execute(f"INSERT INTO {TABLE} BY NAME {select}", [partition, source_path])
        rows_out = con.execute(
            f"SELECT count(*) FROM {TABLE} WHERE source_partition = ?", [partition]
        ).fetchone()[0]
        if rows_out != rows_in:
            raise RuntimeError(
                f"row count mismatch for {partition}: read {rows_in}, loaded {rows_out}"
            )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    log.info("partition %s: rows in %d, rows loaded %d", partition, rows_in, rows_out)
    return rows_out


def load_partition(partition: str, source_path, db_conn=None) -> int:
    """Replace the rows of one partition in raw_trips. Returns the rows loaded.

    db_conn: None (use DB_CONN), a connection URL string, or an open connection."""
    if partition not in config.PARTITIONS:
        raise ValueError(f"partition {partition!r} not in {config.PARTITIONS}")
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    if isinstance(db_conn, duckdb.DuckDBPyConnection):
        return _load(db_conn, partition, str(source))
    if db_conn is None:
        con = db.connect(read_only=False)
    else:
        con = duckdb.connect(db.duckdb_path(db_conn))
    try:
        return _load(con, partition, str(source))
    finally:
        con.close()


def _project_root() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "pyproject.toml").exists():
            return folder
    raise RuntimeError("project root (pyproject.toml) not found")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Load one partition into raw_trips.")
    parser.add_argument("partition")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.partition not in config.PARTITIONS:
        parser.error(f"partition {args.partition!r} not in {config.PARTITIONS}")
    root = _project_root()
    name = config.DOWNLOAD_FILE_PATTERN.format(partition=args.partition)
    source = root / config.STAGING_DIR / name
    verify_source_hash(source, root / "tests" / "reference" / "source_sha256.txt")
    load_partition(args.partition, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
