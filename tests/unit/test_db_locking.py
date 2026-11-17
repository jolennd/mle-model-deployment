"""One writer at a time: connect() is read-only by default and says clearly when the file is locked."""

import subprocess
import sys
from contextlib import contextmanager

import duckdb
import pytest

from taxi_duration.db import connect

HOLD_WRITE_LOCK = (
    "import duckdb, sys\n"
    "con = duckdb.connect(sys.argv[1])\n"
    "con.execute('CREATE TABLE IF NOT EXISTS t (x INTEGER)')\n"
    "print('ready', flush=True)\n"
    "sys.stdin.read()\n"
)


@contextmanager
def other_process_holds_writer(path):
    """A second Python process that keeps a write connection open until the block ends."""
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", HOLD_WRITE_LOCK, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        yield
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)


def make_database(path):
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE t (x INTEGER)")
    con.execute("INSERT INTO t VALUES (1)")
    con.close()


def test_connect_is_read_only_by_default(db_conn):
    make_database(db_conn)
    con = connect()
    try:
        assert con.execute("SELECT x FROM t").fetchone() == (1,)
        with pytest.raises(duckdb.Error):
            con.execute("INSERT INTO t VALUES (2)")
    finally:
        con.close()


def test_read_only_connect_works_when_no_writer_is_open(db_conn):
    make_database(db_conn)
    first = connect()
    second = connect()
    try:
        assert first.execute("SELECT count(*) FROM t").fetchone() == (1,)
        assert second.execute("SELECT count(*) FROM t").fetchone() == (1,)
    finally:
        first.close()
        second.close()


def test_explicit_write_connection_still_writes(db_conn):
    make_database(db_conn)
    con = connect(read_only=False)
    con.execute("INSERT INTO t VALUES (2)")
    con.close()
    check = connect()
    try:
        assert check.execute("SELECT count(*) FROM t").fetchone() == (2,)
    finally:
        check.close()


@pytest.mark.parametrize("read_only", [True, False])
def test_locked_file_gives_a_clear_message(db_conn, read_only):
    make_database(db_conn)
    with other_process_holds_writer(db_conn):
        with pytest.raises(RuntimeError) as err:
            connect(read_only=read_only)
    message = str(err.value)
    assert f"{db_conn.name} is locked by another process" in message
    assert "close notebooks, the API and running loads, then retry" in message
