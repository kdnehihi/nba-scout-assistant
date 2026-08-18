from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.schemas.forecasting import PredictLongTermRequest, PredictShortTermRequest
from app.schemas.recommendation import RecommendationRequest
from app.schemas.scouting import ScoutingReportRequest
from app.services.forecast_service import (
    build_long_term_forecast_response,
    build_short_term_forecast_response,
)
from app.services.recommendation_service import build_recommendation_response
from app.services.reporting_service import build_metadata_response, build_scouting_report_response
from app.services.resources import AppResources, load_app_resources


app = FastAPI(title="NBA Scout Assistant")


def get_resources() -> AppResources:
    """Return loaded API resources from application state."""
    return app.state.resources


@app.on_event("startup")
def startup() -> None:
    app.state.resources = load_app_resources(data_dir="data", artifact_dir="artifacts")


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
