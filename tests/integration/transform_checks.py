"""dbt adapter for test_transform_checks.py. dbt runs its tests as one batch, so there is one
case: it runs `dbt test` through the bridge and returns the number of failed dbt tests.
Run it through make (make loads .env, which sets DB_CONN)."""

import re
import subprocess
import sys
from pathlib import Path

BRIDGE = Path(__file__).resolve().parents[2] / "transform" / "dbt_env.py"


def cases() -> list[str]:
    return ["dbt test"]


def bad_rows(name: str) -> int:
    r = subprocess.run(
        [sys.executable, str(BRIDGE), "test"],
        check=False,
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        return 0
    print(r.stdout[-4000:], r.stderr[-2000:])
    counts = re.findall(r"(?:ERROR|FAIL)=(\d+)", r.stdout)
    return sum(int(c) for c in counts) or 1
