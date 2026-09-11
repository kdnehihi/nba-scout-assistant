from __future__ import annotations

from typing import Any

from app.schemas.forecasting import PredictLongTermRequest, PredictShortTermRequest
from app.services.cache import (
    build_cache_key,
    cache_ttl_seconds,
    get_or_set_cached_json,
)
from app.services.resources import AppResources
from src.api_utils import json_safe
from src.pipelines.forecasting import predict_long_term_tasks, predict_short_term_tasks


def build_short_term_forecast_response(
    resources: AppResources,
    request: PredictShortTermRequest,
) -> dict[str, Any]:
    """Return short-term forecast predictions for requested tasks."""
    key = build_cache_key(
        "short-term-forecast",
        resources.runtime_version,
        request.model_dump(mode="json"),
    )

    def build() -> dict[str, Any]:
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

    return get_or_set_cached_json(
        resources.response_cache,
        key,
        cache_ttl_seconds("FORECAST_CACHE_TTL_SECONDS", 3600),
        build,
    )


def build_long_term_forecast_response(
    resources: AppResources,
    request: PredictLongTermRequest,
) -> dict[str, Any]:
    """Return long-term forecast predictions for requested tasks and horizons."""
    key = build_cache_key(
        "long-term-forecast",
        resources.runtime_version,
        request.model_dump(mode="json"),
    )

    def build() -> dict[str, Any]:
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

    return get_or_set_cached_json(
        resources.response_cache,
        key,
        cache_ttl_seconds("FORECAST_CACHE_TTL_SECONDS", 3600),
        build,
    )
