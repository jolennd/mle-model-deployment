from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from taxi_duration import config

MAX_ZONE_ID = 265


def _check_zone(value: int) -> int:
    if value in config.EXCLUDED_ZONE_IDS:
        raise ValueError("zone id is not served")
    return value


class PredictRequest(BaseModel):
    pickup_zone_id: int = Field(ge=1, le=MAX_ZONE_ID)
    dropoff_zone_id: int = Field(ge=1, le=MAX_ZONE_ID)
    trip_distance: float = Field(
        ge=config.MIN_DISTANCE_MILES, le=config.API_MAX_DISTANCE_MILES
    )
    pickup_at: datetime

    @field_validator("pickup_zone_id", "dropoff_zone_id")
    @classmethod
    def zone_is_served(cls, value: int) -> int:
        return _check_zone(value)


class PredictResponse(BaseModel):
    prediction_minutes: float
    model_version: str


class HealthResponse(BaseModel):
    ready: bool
