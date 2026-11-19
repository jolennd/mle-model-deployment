"""Training sample: fixed seed, sorted by trip_id, independent of input row order."""

import numpy as np
import pandas as pd

from taxi_duration import config


def make_frame(n=1000):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "trip_id": [f"t{i:05d}" for i in range(n)],
            "trip_distance": rng.uniform(0.5, 10, n),
        }
    )
    return df.sample(frac=1, random_state=3).reset_index(drop=True)  # shuffled


def test_sample_fixed_seed():
    from taxi_duration.splits import draw_sample

    df = make_frame()
    first = draw_sample(df, size=100, seed=config.SEED)
    again = draw_sample(df, size=100, seed=config.SEED)
    other = draw_sample(df, size=100, seed=config.SEED + 1)

    ids = first["trip_id"].tolist()
    assert ids == again["trip_id"].tolist()
    assert ids != other["trip_id"].tolist()
    assert len(first) == 100
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)

    # input row order does not matter
    reordered = df.sample(frac=1, random_state=11).reset_index(drop=True)
    assert draw_sample(reordered, size=100, seed=config.SEED)["trip_id"].tolist() == ids

    # fewer rows than the size: all rows, sorted
    small = draw_sample(df.head(30), size=100, seed=config.SEED)
    assert small["trip_id"].tolist() == sorted(df.head(30)["trip_id"])
