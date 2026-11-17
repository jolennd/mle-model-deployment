"""Database access for the duckdb variant. One helper; no ORM."""

from pathlib import Path
from urllib.parse import urlparse

import duckdb

from .config import require_env


def duckdb_path(url: str) -> str:
    """duckdb:///rel.duckdb is relative to the project root, duckdb:////abs.duckdb is absolute.
    Same rule as the transform bridge (tests/unit/test_db_path_matches_transform.py)."""
    parsed = urlparse(url)
    if parsed.scheme != "duckdb":
        raise ValueError(f"unsupported DB_CONN scheme: {parsed.scheme!r}")
    path = parsed.path[1:] if parsed.path.startswith("//") else parsed.path.lstrip("/")
    if not path:
        raise ValueError("DB_CONN has no database path")
    return path


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the database named by DB_CONN. One writer at a time.

    Read-only by default. Code that writes (the load step) must pass read_only=False.
    A lock held by another process raises RuntimeError with a plain message."""
    path = duckdb_path(require_env("DB_CONN"))
    try:
        return duckdb.connect(path, read_only=read_only)
    except duckdb.IOException as err:
        if "lock" not in str(err).lower():
            raise
        raise RuntimeError(
            f"{Path(path).name} is locked by another process: "
            "close notebooks, the API and running loads, then retry."
        ) from err
