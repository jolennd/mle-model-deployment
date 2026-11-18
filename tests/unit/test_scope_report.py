"""Scope report (ADR 7): per rule, rows removed alone and rows added after earlier rules."""

import pandas as pd

from taxi_duration.labels import scope_report

T0 = pd.Timestamp("2024-01-15 12:00:00")
EXCLUDED = 264


def row(trip_id, minutes=10.0, miles=2.0, pickup_zone=100, dropoff_zone=100):
    return {
        "trip_id": trip_id,
        "pickup_at": T0,
        "dropoff_at": T0 + pd.Timedelta(minutes=minutes),
        "pickup_zone_id": pickup_zone,
        "dropoff_zone_id": dropoff_zone,
        "trip_distance": miles,
    }


def crafted_frame():
    return pd.DataFrame(
        [
            row("ok"),
            row("dur", minutes=0.5, miles=0.5),
            row("dur_dist", minutes=0.5, miles=0.05),  # duration and distance
            row("dist", miles=0.05),
            row("speed", miles=20.0),
            row("speed_zone", miles=20.0, pickup_zone=EXCLUDED),  # speed and zones
            row("zone", dropoff_zone=EXCLUDED),
            row("late", minutes=20.0),  # ends after the embargo limit
            row("late_zone", minutes=20.0, pickup_zone=EXCLUDED),  # zones and embargo
        ]
    )


def test_scope_report_counts():
    df = crafted_frame()
    limit = T0 + pd.Timedelta(minutes=15)

    with_embargo = scope_report(df, dropoff_before=limit)
    assert with_embargo["input"] == 9
    assert with_embargo["duration"] == {"alone": 2, "added": 2}
    assert with_embargo["distance"] == {"alone": 2, "added": 1}
    assert with_embargo["speed"] == {"alone": 2, "added": 2}
    assert with_embargo["zones"] == {"alone": 3, "added": 2}
    assert with_embargo["embargo"] == {"alone": 2, "added": 1}
    added = sum(
        with_embargo[r]["added"]
        for r in ("duration", "distance", "speed", "zones", "embargo")
    )
    assert with_embargo["kept"] == with_embargo["input"] - added == 1

    without = scope_report(df)
    assert without["embargo"] == {"alone": 0, "added": 0}
    assert without["duration"] == with_embargo["duration"]
    assert without["zones"] == with_embargo["zones"]
    assert without["kept"] == 2  # "ok" and "late"
