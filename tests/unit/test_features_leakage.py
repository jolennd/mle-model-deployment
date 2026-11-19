"""Features never use columns that exist only after the trip (ADR 2)."""

import pandas as pd

from taxi_duration import config


def test_post_event_columns_never_features():
    post = {name.lower() for name in config.POST_TRIP_COLUMNS}
    assert not [c for c in config.FEATURE_COLUMNS if c.lower() in post]

    from taxi_duration.features import build_features

    row = {
        "trip_id": "a",
        "pickup_at": pd.Timestamp("2025-01-10 08:15:00"),
        "dropoff_at": pd.Timestamp("2025-01-10 08:30:00"),
        "pickup_zone_id": 100,
        "dropoff_zone_id": 200,
        "trip_distance": 2.5,
    }
    for i, name in enumerate(config.POST_TRIP_COLUMNS):
        row[name] = 1.0 + i
        row[name.upper() + "_X"] = 2.0  # look-alike names are ignored too
    row["FARE_AMOUNT"] = 9.9
    row["Tip_Amount"] = 1.1
    out = build_features(pd.DataFrame([row]))

    assert list(out.columns) == list(config.FEATURE_COLUMNS)
    assert not [c for c in out.columns if c.lower() in post]
