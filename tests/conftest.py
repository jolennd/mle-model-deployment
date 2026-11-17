import pytest


@pytest.fixture
def db_conn(monkeypatch, tmp_path):
    """DB_CONN pointing at a fresh database file in a temporary folder. Never the real database."""
    monkeypatch.setenv("DB_CONN", f"duckdb:///{tmp_path / 'test.duckdb'}")
    return tmp_path / "test.duckdb"
