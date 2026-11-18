import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HealthResponse, PredictRequest, PredictResponse
from taxi_duration import verify_model
from taxi_duration.features import build_features

logger = logging.getLogger("app")

NOT_READY_MESSAGE = "Model not available"


def _load_model(state: dict) -> None:
    """Read settings from the environment and load the verified model once."""
    model_uri = os.environ.get("MODEL_URI", "")
    expected = os.environ.get("MODEL_SHA256", "")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    trusted = frozenset(
        t.strip()
        for t in os.environ.get("MODEL_TRUSTED_TYPES", "").split(",")
        if t.strip()
    )
    try:
        verify_model.check_strict_tracking_uri(tracking_uri)
        version = verify_model.resolve_version(model_uri, tracking_uri)
        model, _ = verify_model.verify_then_load(
            model_uri,
            expected_sha256=expected,
            tracking_uri=tracking_uri,
            trusted_types=trusted,
            version=version,
        )
    except Exception as exc:  # noqa: BLE001 - any load failure means not ready
        # Log only the error type: messages can hold the hash or file paths.
        logger.error("Model could not be loaded (%s)", type(exc).__name__)
        state.update(model=None, version=None, ready=False)
        return
    state.update(model=model, version=str(version), ready=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state: dict = {"model": None, "version": None, "ready": False}
    _load_model(state)
    app.state.model_state = state
    yield


app = FastAPI(title="Taxi duration", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ready=bool(app.state.model_state["ready"]))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    state = app.state.model_state
    if not state["ready"]:
        raise HTTPException(status_code=503, detail=NOT_READY_MESSAGE)
    row = pd.DataFrame(
        [
            {
                "trip_distance": request.trip_distance,
                "pickup_at": request.pickup_at,
                "pickup_zone_id": request.pickup_zone_id,
                "dropoff_zone_id": request.dropoff_zone_id,
            }
        ]
    )
    try:
        value = float(state["model"].predict(build_features(row))[0])
    except Exception:  # noqa: BLE001 - never show model errors to the caller
        logger.error("Prediction failed")
        raise HTTPException(status_code=503, detail=NOT_READY_MESSAGE) from None
    return PredictResponse(prediction_minutes=value, model_version=state["version"])
