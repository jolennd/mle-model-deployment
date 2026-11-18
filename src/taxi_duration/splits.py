"""The one door to the rows.

EDA and fitting read rows only through load_train(). load_test() refuses until
the baseline lock exists.
"""

import json
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


def draw_sample(
    df: pd.DataFrame, size: int = config.SAMPLE_SIZE, seed: int = config.SEED
) -> pd.DataFrame:
    """Seeded sample of `size` rows, sorted by trip_id. Row order of the input is irrelevant."""
    ordered = df.sort_values("trip_id").reset_index(drop=True)
    if len(ordered) > size:
        ordered = ordered.sample(n=size, random_state=seed)
    return ordered.sort_values("trip_id").reset_index(drop=True)


def make_folds(df: pd.DataFrame) -> list[dict]:
    """Expanding time folds by pickup day (docs/tuning-plan.md).

    Valid: pickups on the validation days. Train: pickups up to the last train
    day that also end before the first validation day starts (embargo).
    """
    day = pd.Timedelta(days=1)
    ids = df["trip_id"]
    folds = []
    for last_train, first_valid, last_valid in config.CV_FOLDS:
        window_start = pd.Timestamp(first_valid)
        window_end = pd.Timestamp(last_valid) + day
        in_valid = (df["pickup_at"] >= window_start) & (df["pickup_at"] < window_end)
        in_train = (df["pickup_at"] < pd.Timestamp(last_train) + day) & (
            df["dropoff_at"] < window_start
        )
        folds.append(
            {
                "train": sorted(ids[in_train].tolist()),
                "valid": sorted(ids[in_valid].tolist()),
            }
        )
    return folds


def write_folds(folds: list[dict], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as handle:
        json.dump(folds, handle)


def read_folds(path: str) -> list[dict]:
    with open(path) as handle:
        return json.load(handle)
