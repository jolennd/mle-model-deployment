"""Fitted steps learn from the training fold only."""

import numpy as np
import pandas as pd
import pytest

from taxi_duration import features


def make_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    pickup = pd.Timestamp("2025-01-05") + pd.to_timedelta(
        rng.integers(0, 15 * 24 * 60, n), unit="m"
    )
    distance = rng.uniform(0.5, 10, n)
    return pd.DataFrame(
        {
            "trip_id": [f"t{i:04d}" for i in range(n)],
            "pickup_at": pickup,
            "pickup_zone_id": rng.choice([10, 20, 30], n),
            "dropoff_zone_id": rng.choice([40, 50], n),
            "trip_distance": distance,
            "duration_minutes": 3 + 2 * distance + rng.normal(0, 0.2, n),
        }
    )


def small_pipeline():
    return features.make_pipeline().set_params(model__n_estimators=5)


def fit(frame):
    pipe = small_pipeline()
    pipe.fit(features.build_features(frame), frame["duration_minutes"])
    return pipe


def encoder_categories(pipe):
    return [
        list(c)
        for c in pipe.named_steps["encode"].named_transformers_["zones"].categories_
    ]


def test_transforms_fit_on_training_fold_only():
    frame = make_frame()
    train, valid = frame.iloc[:150], frame.iloc[150:]
    changed = valid.assign(
        pickup_zone_id=999,
        dropoff_zone_id=888,
        trip_distance=valid["trip_distance"] * 100,
    )

    alone = encoder_categories(fit(train))
    # the encoder is fitted on the train frame only; valid rows never reach it
    assert alone == [[10, 20, 30], [40, 50]]
    pipe = fit(train)
    assert encoder_categories(pipe) == alone

    # a zone id seen only in valid rows is unknown to the fitted encoder
    encoded = pipe.named_steps["encode"].transform(features.build_features(changed))
    assert (encoded[:, :2] == -1).all()
    assert encoder_categories(pipe) == alone  # transform did not learn it

    # cross_validate: scaling valid targets changes scores, not the fold-1 fit state
    from taxi_duration import train as train_module

    ids = frame["trip_id"].tolist()
    folds = [
        {"train": ids[:100], "valid": ids[100:130]},
        {"train": ids[:130], "valid": ids[130:160]},
    ]

    def run(df):
        captured = []

        def factory():
            pipe = small_pipeline()
            captured.append(pipe)
            return pipe

        scores = train_module.cross_validate(df, folds, pipeline_factory=factory)
        return scores, captured

    base_scores, base_pipes = run(frame)
    scaled = frame.copy()
    valid_ids = set(folds[0]["valid"])
    mask = scaled["trip_id"].isin(valid_ids)
    scaled.loc[mask, "duration_minutes"] *= 100
    scaled_scores, scaled_pipes = run(scaled)

    assert scaled_scores["model"][0] != pytest.approx(base_scores["model"][0])
    assert encoder_categories(scaled_pipes[0]) == encoder_categories(base_pipes[0])
    assert encoder_categories(base_pipes[0]) == encoder_categories(
        fit(frame[frame["trip_id"].isin(folds[0]["train"])])
    )
