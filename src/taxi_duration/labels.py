"""Label and scope rules (ADR 7).

One code path for training rows and test rows. The speed rule uses the label,
so it only defines valid records and must never be applied to a request.
"""

import pandas as pd

from taxi_duration import config

RULES = ("duration", "distance", "speed", "zones", "embargo")


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with float duration_minutes = dropoff_at - pickup_at."""
    out = df.copy()
    delta = out["dropoff_at"] - out["pickup_at"]
    out["duration_minutes"] = (delta / pd.Timedelta(minutes=1)).astype(float)
    return out


def _removed_masks(
    labelled: pd.DataFrame, dropoff_before: pd.Timestamp | str | None
) -> dict[str, pd.Series]:
    """Per rule, True where the rule removes the row. Missing values are removed."""
    minutes = labelled["duration_minutes"]
    miles = labelled["trip_distance"]
    keep = {
        "duration": minutes.between(
            config.MIN_DURATION_MINUTES, config.MAX_DURATION_MINUTES
        ),
        "distance": miles.between(config.MIN_DISTANCE_MILES, config.MAX_DISTANCE_MILES),
        # miles / (minutes / 60) <= cap, written without a division so a zero
        # duration gives no divide error (such rows fail the speed rule or are
        # already removed by the duration rule)
        "speed": miles * 60 <= config.MAX_SPEED_MPH * minutes,
        "zones": ~(
            labelled["pickup_zone_id"].isin(config.EXCLUDED_ZONE_IDS)
            | labelled["dropoff_zone_id"].isin(config.EXCLUDED_ZONE_IDS)
        ),
    }
    if dropoff_before is None:
        keep["embargo"] = pd.Series(True, index=labelled.index)
    else:
        keep["embargo"] = labelled["dropoff_at"] < pd.Timestamp(dropoff_before)
    return {name: ~keep[name].fillna(False).astype(bool) for name in RULES}


def apply_scope(
    df: pd.DataFrame, dropoff_before: pd.Timestamp | str | None = None
) -> pd.DataFrame:
    """Add duration_minutes and keep rows that satisfy all rules of ADR 7."""
    labelled = add_label(df)
    removed = pd.Series(False, index=labelled.index)
    for mask in _removed_masks(labelled, dropoff_before).values():
        removed |= mask
    return labelled.loc[~removed]


def scope_report(
    df: pd.DataFrame, dropoff_before: pd.Timestamp | str | None = None
) -> dict:
    """Row counts per rule: alone, and added after the earlier rules in order."""
    labelled = add_label(df)
    masks = _removed_masks(labelled, dropoff_before)
    report: dict = {"input": len(labelled)}
    removed_so_far = pd.Series(False, index=labelled.index)
    for name in RULES:
        mask = masks[name]
        report[name] = {
            "alone": int(mask.sum()),
            "added": int((mask & ~removed_so_far).sum()),
        }
        removed_so_far |= mask
    report["kept"] = int((~removed_so_far).sum())
    return report
