"""Reference numbers: results that must not change when dependencies are upgraded.

Each tests/reference/<name>.json holds checks recorded from a known good run:

    {
      "description": "Partition 2025-01 after make pipeline-run PARTITION=2025-01",
      "checks": [
        {"name": "rows_mart", "sql": "select count(*) from mart_daily_revenue", "value": 31},
        {"name": "revenue", "sql": "select sum(revenue) from mart_daily_revenue",
         "value": 1234.56, "tolerance": 0.01}
      ]
    }

Write the checks (name and sql), then fill the values from the current database with
`make reference-record`. `make upgrade-check` rebuilds and compares. The tests need the
built database, so they carry the `reference` marker and are not part of test-fast.
"""

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

REF_DIR = Path(__file__).resolve().parents[1] / "reference"
FILES = sorted(REF_DIR.glob("*.json"))


def scalar(sql: str):
    """One value from the project database, read only. Works with either db.py shape."""
    from taxi_duration import db

    if hasattr(db, "connect"):  # file database helper
        con = db.connect(read_only=True)
        try:
            return con.execute(sql).fetchone()[0]
        finally:
            con.close()
    from sqlalchemy import text  # server database helper

    with db.make_engine().connect() as c:
        return c.execute(text(sql)).scalar()


def _cases():
    for f in FILES:
        for check in json.loads(f.read_text())["checks"]:
            yield pytest.param(check, id=f"{f.stem}:{check['name']}")


@pytest.mark.reference
@pytest.mark.parametrize("check", list(_cases()))
def test_reference_number_unchanged(check):
    if "value" not in check:
        pytest.fail(f"{check['name']} has no recorded value: run make reference-record")
    got = scalar(check["sql"])
    tol = check.get("tolerance", 0)
    assert got is not None, f"{check['name']}: query returned NULL"
    assert abs(float(got) - float(check["value"])) <= tol, (
        f"{check['name']}: expected {check['value']}, got {got}"
    )


def record() -> None:
    """Fill or refresh every value from the current database."""
    if not FILES:
        sys.exit(f"no reference files in {REF_DIR}: write the checks first")
    for f in FILES:
        data = json.loads(f.read_text())
        for check in data["checks"]:
            value = scalar(check["sql"])
            check["value"] = float(value) if isinstance(value, Decimal) else value
            print(f"{f.stem}:{check['name']} = {check['value']}")
        f.write_text(json.dumps(data, indent=2, default=str) + "\n")


if __name__ == "__main__" and "--record" in sys.argv:
    record()
