"""A model is registered only when its file is strictly under the size limit."""

import pytest

from taxi_duration import config


@pytest.mark.parametrize(
    ("size", "allowed"),
    [
        (config.MODEL_SIZE_LIMIT_BYTES - 1, True),
        (1, True),
        (config.MODEL_SIZE_LIMIT_BYTES, False),
        (config.MODEL_SIZE_LIMIT_BYTES + 1, False),
    ],
)
def test_model_registered_only_under_size_limit(size, allowed):
    from taxi_duration import train

    calls = []
    result = train.register_if_allowed(size, lambda: calls.append(1))

    assert result is allowed
    assert len(calls) == (1 if allowed else 0)
