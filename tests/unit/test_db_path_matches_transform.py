"""db.duckdb_path and the transform bridge must resolve DB_CONN to the same file."""

import importlib.util
from pathlib import Path

import pytest

from taxi_duration.db import duckdb_path

BRIDGE = Path(__file__).resolve().parents[2] / "transform" / "dbt_env.py"


@pytest.mark.skipif(not BRIDGE.exists(), reason="no transform bridge in this project")
@pytest.mark.parametrize(
    "url", ["duckdb:///data/warehouse.duckdb", "duckdb:////tmp/abs/warehouse.duckdb"]
)
def test_db_path_matches_transform(url):
    spec = importlib.util.spec_from_file_location("dbt_env", BRIDGE)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    assert duckdb_path(url) == bridge.duckdb_path(url)
