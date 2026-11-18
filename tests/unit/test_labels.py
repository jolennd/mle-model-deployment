"""Label and scope rules (ADR 7): one function for training and test rows, bounds inclusive."""

import pandas as pd

from taxi_duration import config
from taxi_duration.labels import add_label, apply_scope

T0 = pd.Timestamp("2024-01-15 12:00:00")
VALID_MINUTES = 10.0
VALID_MILES = 2.0
VALID_ZONE = 100


def make_row(
    trip_id,
    minutes=VALID_MINUTES,
    miles=VALID_MILES,
    pickup_zone=VALID_ZONE,
    dropoff_zone=VALID_ZONE,
):
    """A valid in-scope trip; tests change one field."""
    return {
        "trip_id": trip_id,
        "pickup_at": T0,
        "dropoff_at": T0 + pd.Timedelta(minutes=minutes),
        "pickup_zone_id": pickup_zone,
        "dropoff_zone_id": dropoff_zone,
        "trip_distance": miles,
    }


def kept_ids(rows):
    out = apply_scope(pd.DataFrame(rows))
    return set(out["trip_id"])


def test_label_definition():
    df = pd.DataFrame(
        [
            {**make_row("a"), "dropoff_at": T0 + pd.Timedelta(seconds=90)},
            {**make_row("b"), "dropoff_at": T0 + pd.Timedelta(minutes=12, seconds=30)},
        ]
    )
    out = add_label(df)
    assert out["duration_minutes"].dtype.kind == "f"
    assert out["duration_minutes"].tolist() == [1.5, 12.5]
    assert "duration_minutes" not in df.columns  # input is not modified


def test_scope_rule_duration():
    lo, hi = config.MIN_DURATION_MINUTES, config.MAX_DURATION_MINUTES
    # tiny distances keep the implied speed far below the cap
    rows = [
        make_row("at_min", minutes=lo, miles=0.5),
        make_row("at_max", minutes=hi, miles=0.5),
        make_row("below_min", minutes=lo - 1 / 60, miles=0.5),
        make_row("above_max", minutes=hi + 1 / 60, miles=0.5),
    ]
    assert kept_ids(rows) == {"at_min", "at_max"}


def test_scope_rule_distance():
    lo, hi = config.MIN_DISTANCE_MILES, config.MAX_DISTANCE_MILES
    # NOTE: 100 miles within 60 minutes is above the speed cap, so the upper
    # bound can only be shown from the removed side (hi + 0.01 is removed by
    # distance, and would be removed by speed too). See report.
    rows = [
        make_row("at_min", miles=lo),
        make_row("below_min", miles=lo - 0.01),
        make_row("zero", miles=0.0),
        make_row("above_max", miles=hi + 0.01),
    ]
    assert kept_ids(rows) == {"at_min"}


def test_scope_rule_speed():
    cap = config.MAX_SPEED_MPH
    minutes = 30.0
    at_cap = cap * minutes / 60  # 40 miles in 30 minutes is exactly 80 mph
    rows = [
        make_row("at_cap", minutes=minutes, miles=at_cap),
        make_row("slower", minutes=minutes, miles=at_cap - 1),
        make_row("above_cap", minutes=minutes, miles=at_cap + 0.5),
    ]
    assert kept_ids(rows) == {"at_cap", "slower"}


def test_scope_rule_zones():
    excluded = config.EXCLUDED_ZONE_IDS[0]
    rows = [
        make_row("pickup_264", pickup_zone=excluded),
        make_row("dropoff_264", dropoff_zone=excluded),
        make_row("both_264", pickup_zone=excluded, dropoff_zone=excluded),
        make_row("zone_265", pickup_zone=265, dropoff_zone=265),
        make_row("zone_1_263", pickup_zone=1, dropoff_zone=263),
    ]
    assert kept_ids(rows) == {"zone_265", "zone_1_263"}
