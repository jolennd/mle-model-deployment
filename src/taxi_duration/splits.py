"""The one door to the rows.

EDA and fitting read rows only through load_train(). load_test() refuses until
the baseline lock exists.
"""

from pathlib import Path

import pandas as pd

from taxi_duration import config, db, labels

_WINDOW_SQL = (
    "SELECT * FROM mart_trips "
    "WHERE pickup_at >= CAST(? AS TIMESTAMP) AND pickup_at < CAST(? AS TIMESTAMP)"
)


def _read(start: str, end: str) -> pd.DataFrame:
    con = db.connect()
    try:
        return con.execute(_WINDOW_SQL, [start, end]).df()
    finally:
        con.close()


def load_train() -> pd.DataFrame:
    """Scoped training rows: pickups from TRAIN_START up to TEST_START.

    The date filter runs in the database, so test-week rows never leave it.
    Trips that drop off at or after TEST_START are removed (embargo).
    """
    df = _read(config.TRAIN_START, config.TEST_START)
    scoped = labels.apply_scope(df, dropoff_before=config.TEST_START)
    return scoped.sort_values("trip_id").reset_index(drop=True)


def load_test() -> pd.DataFrame:
    """Scoped test-week rows. Only evaluation code may call this.

    Raises until the baseline lock file exists. The check runs before the
    database is opened.
    """
    lock = Path(config.LOCK_PATH)
    if not lock.exists():
        raise RuntimeError(
            f"baseline lock not found at {lock}: the test rows stay closed until the baseline is locked"
        )
    df = _read(config.TEST_START, config.TEST_END)
    scoped = labels.apply_scope(df)
    return scoped.sort_values("trip_id").reset_index(drop=True)
