"""Serving-time features and the model pipeline (ADR 2, ADR 8).

build_features is pure: no fitted state, no I/O. The encoder lives in the
pipeline and is fitted only when the pipeline is fitted, so a fold fit sees
training rows only.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from taxi_duration import config


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly config.FEATURE_COLUMNS, in order. Other columns are dropped."""
    pickup = pd.to_datetime(df["pickup_at"])
    out = pd.DataFrame(
        {
            "trip_distance": df["trip_distance"].astype(float),
            "pickup_hour": pickup.dt.hour.astype(int),
            "pickup_dow": pickup.dt.day_of_week.astype(int),
            "pickup_zone_id": df["pickup_zone_id"],
            "dropoff_zone_id": df["dropoff_zone_id"],
        },
        index=df.index,
    )
    return out[list(config.FEATURE_COLUMNS)]


def make_pipeline(seed: int = config.SEED) -> Pipeline:
    """Unfitted pipeline: ordinal encoding of zone ids, then a random forest."""
    categorical = list(config.CATEGORICAL_FEATURES)
    encode = ColumnTransformer(
        [
            (
                "zones",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("encode", encode),
            ("model", RandomForestRegressor(random_state=seed)),
        ]
    )
