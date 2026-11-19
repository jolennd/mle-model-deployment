"""Batch 2 label tests: the embargo rule (5) and identical treatment of train and test rows (ADR 7)."""

import pandas as pd

from taxi_duration import config
from taxi_duration.labels import apply_scope

CUTOFF = pd.Timestamp(config.TEST_START)


def trip(trip_id, pickup, minutes=10.0, miles=2.0, pickup_zone=100, dropoff_zone=100):
    pickup = pd.Timestamp(pickup)
    return {
        "trip_id": trip_id,
        "pickup_at": pickup,
        "dropoff_at": pickup + pd.Timedelta(minutes=minutes),
        "pickup_zone_id": pickup_zone,
        "dropoff_zone_id": dropoff_zone,
        "trip_distance": miles,
    }


def test_scope_rule_embargo():
    exact = {**trip("exact", CUTOFF - pd.Timedelta(minutes=10)), "dropoff_at": CUTOFF}
    before = {
        **trip("before", CUTOFF - pd.Timedelta(minutes=11)),
        "dropoff_at": CUTOFF - pd.Timedelta(minutes=1),
    }
    df = pd.DataFrame([exact, before])

    kept = apply_scope(df, dropoff_before=CUTOFF)
    assert set(kept["trip_id"]) == {"before"}

    # without a cutoff this rule removes nothing
    assert set(apply_scope(df, dropoff_before=None)["trip_id"]) == {"exact", "before"}
    assert set(apply_scope(df)["trip_id"]) == {"exact", "before"}


def test_scope_same_for_train_and_test():
    excluded = config.EXCLUDED_ZONE_IDS[0]
    df = pd.DataFrame(
        [
            trip("ok_1", "2025-01-10 08:00", minutes=12.5),
            trip("ok_2", "2025-01-20 17:30", minutes=45.0, miles=9.0),
            trip("too_short", "2025-01-11 09:00", minutes=0.5),
            trip("too_long", "2025-01-11 10:00", minutes=61.0, miles=1.0),
            trip("too_near", "2025-01-12 09:00", miles=0.05),
            trip("too_fast", "2025-01-12 10:00", minutes=30.0, miles=41.0),
            trip("zone_264", "2025-01-13 09:00", dropoff_zone=excluded),
            trip("ok_late", "2025-01-26 12:00", minutes=20.0),
        ]
    )
    later_than_all = pd.Timestamp("2025-02-01")

    train_way = apply_scope(df, dropoff_before=later_than_all)
    test_way = apply_scope(df)

    # a cutoff later than every dropoff changes nothing
    assert list(train_way["trip_id"]) == list(test_way["trip_id"])
    assert set(test_way["trip_id"]) == {"ok_1", "ok_2", "ok_late"}
    assert (
        train_way["duration_minutes"].tolist() == test_way["duration_minutes"].tolist()
    )

    # the embargo only removes rows; rows it keeps are identical to the test way
    embargoed = apply_scope(df, dropoff_before=CUTOFF)
    assert set(embargoed["trip_id"]) == {"ok_1", "ok_2"}
    both = test_way.set_index("trip_id").loc[["ok_1", "ok_2"], "duration_minutes"]
    assert embargoed.set_index("trip_id")["duration_minutes"].tolist() == both.tolist()

    # rules 1 to 4 give the same verdict for a row alone or inside a frame
    in_frame = set(test_way["trip_id"])
    for i in range(len(df)):
        one = apply_scope(df.iloc[[i]])
        assert (len(one) == 1) == (df.iloc[i]["trip_id"] in in_frame)
