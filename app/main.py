from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.schemas.forecasting import PredictLongTermRequest, PredictShortTermRequest
from app.schemas.recommendation import RecommendationRequest
from app.schemas.scouting import ScoutingReportRequest
from app.services.forecast_service import (
    build_long_term_forecast_response,
    build_short_term_forecast_response,
)
from app.services.recommendation_service import build_recommendation_response
from app.services.reporting_service import (
    build_metadata_response,
    build_scouting_report_response,
)
from app.services.resources import AppResources, load_app_resources

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared data and model artifacts before accepting API requests."""
    app.state.resources = load_app_resources(data_dir="data", artifact_dir="artifacts")
    yield


app = FastAPI(title="NBA Scout Assistant", lifespan=lifespan)


def get_resources() -> AppResources:
    """Return loaded API resources from application state."""
    return app.state.resources


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return build_metadata_response(get_resources())


@app.post("/recommendations")
def recommendations(request: RecommendationRequest):
    try:
        return build_recommendation_response(get_resources(), request)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/forecasts/short-term")
def short_term_forecast(request: PredictShortTermRequest):
    try:
        return build_short_term_forecast_response(get_resources(), request)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/forecasts/long-term")
def long_term_forecast(request: PredictLongTermRequest):
    try:
        return build_long_term_forecast_response(get_resources(), request)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/players/scouting-report")
def scouting_report(request: ScoutingReportRequest):
    try:
        return build_scouting_report_response(get_resources(), request)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
