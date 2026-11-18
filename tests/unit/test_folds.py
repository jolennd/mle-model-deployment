"""CV folds (docs/tuning-plan.md): expanding windows by pickup day, embargo before each window."""

import json

import pandas as pd

from taxi_duration import config


def make_frame():
    """Three trips per day for January: 02:00, 12:00, and 23:50 ending after midnight."""
    rows = []
    n = 0
    for day in pd.date_range("2025-01-01", "2025-01-31", freq="D"):
        for offset, minutes in ((2 * 60, 10), (12 * 60, 10), (23 * 60 + 50, 30)):
            pickup = day + pd.Timedelta(minutes=offset)
            rows.append(
                {
                    "trip_id": f"t{n:04d}",
                    "pickup_at": pickup,
                    "dropoff_at": pickup + pd.Timedelta(minutes=minutes),
                    "pickup_zone_id": 100,
                    "dropoff_zone_id": 100,
                    "trip_distance": 2.0,
                }
            )
            n += 1
    return pd.DataFrame(rows)


def folds_for(df):
    from taxi_duration.splits import make_folds

    return make_folds(df)


def test_folds_round_trip(tmp_path):
    from taxi_duration.splits import read_folds, write_folds

    folds = folds_for(make_frame())
    path = tmp_path / "cv_folds.json"
    write_folds(folds, str(path))

    with open(path) as handle:
        assert json.load(handle) == folds  # valid JSON, same structure
    assert read_folds(str(path)) == folds


def test_folds_respect_embargo():
    df = make_frame().set_index("trip_id")
    folds = folds_for(df.reset_index())

    for fold, (last_train, first_valid, last_valid) in zip(
        folds, config.CV_FOLDS, strict=True
    ):
        window_start = pd.Timestamp(first_valid)
        window_end = pd.Timestamp(last_valid) + pd.Timedelta(days=1)
        train = df.loc[fold["train"]]
        valid = df.loc[fold["valid"]]

        assert len(train) > 0
        assert len(valid) > 0
        assert (train["dropoff_at"] < window_start).all()
        assert (
            train["pickup_at"] < pd.Timestamp(last_train) + pd.Timedelta(days=1)
        ).all()
        assert (valid["pickup_at"] >= window_start).all()
        assert (valid["pickup_at"] < window_end).all()
        # every trip picked up inside the window is validated
        in_window = df.index[
            (df["pickup_at"] >= window_start) & (df["pickup_at"] < window_end)
        ]
        assert set(fold["valid"]) == set(in_window)
        assert set(fold["train"]).isdisjoint(fold["valid"])
        assert fold["train"] == sorted(fold["train"])
        assert fold["valid"] == sorted(fold["valid"])

        # the trip that starts at 23:50 on the last train day ends after midnight
        straddle = df.index[
            df["pickup_at"]
            == pd.Timestamp(last_train) + pd.Timedelta(hours=23, minutes=50)
        ]
        assert len(straddle) == 1
        assert straddle[0] not in set(fold["train"])


def test_folds_expand_in_time_order():
    df = make_frame()
    folds = folds_for(df)
    assert len(folds) == len(config.CV_FOLDS) == 3

    pickup = df.set_index("trip_id")["pickup_at"]
    previous_train: set = set()
    previous_end = None
    for fold in folds:
        train = set(fold["train"])
        assert previous_train <= train
        previous_train = train

        valid_days = pickup.loc[fold["valid"]].dt.normalize()
        start, end = valid_days.min(), valid_days.max()
        if previous_end is not None:
            assert start > previous_end
        previous_end = end
