from __future__ import annotations

from typing import Any

from app.schemas.forecasting import PredictLongTermRequest, PredictShortTermRequest
from app.services.resources import AppResources
from src.api_utils import json_safe
from src.pipelines.forecasting import predict_long_term_tasks, predict_short_term_tasks


def build_short_term_forecast_response(
    resources: AppResources,
    request: PredictShortTermRequest,
) -> dict[str, Any]:
    """Return short-term forecast predictions for requested tasks."""
    predictions = predict_short_term_tasks(
        performance_df=resources.short_term_data,
        artifacts=resources.short_term_models,
        tasks=request.tasks,
        player_id=request.player_id,
        player_name=request.player_name,
        season=request.season,
        as_of_date=request.as_of_date,
    )
    return json_safe({"predictions": predictions})


def build_long_term_forecast_response(
    resources: AppResources,
    request: PredictLongTermRequest,
) -> dict[str, Any]:
    """Return long-term forecast predictions for requested tasks and horizons."""
    predictions = predict_long_term_tasks(
        long_term_df=resources.long_term_data,
        artifacts=resources.long_term_models,
        tasks=request.tasks,
        horizons=request.horizons,
        player_id=request.player_id,
        player_name=request.player_name,
        anchor_season=request.anchor_season,
    )
    return json_safe({"predictions": predictions})
