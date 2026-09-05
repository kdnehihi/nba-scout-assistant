from __future__ import annotations

from typing import Any

from app.schemas.scouting import ScoutingReportRequest
from app.services.resources import AppResources
from src.api_utils import json_safe
from src.pipelines.reporting import build_player_scouting_report, build_service_metadata


def build_metadata_response(resources: AppResources) -> dict[str, Any]:
    """Return API metadata for loaded data and model artifacts."""
    return json_safe(
        build_service_metadata(
            recommendation_data=resources.recommendation_data,
            performance_df=resources.short_term_data,
            long_term_df=resources.long_term_data,
            short_term_models=resources.short_term_models,
            long_term_models=resources.long_term_models,
        )
    )


def build_scouting_report_response(
    resources: AppResources,
    request: ScoutingReportRequest,
) -> dict[str, Any]:
    """Return player profile, compensation context, signals, and forecasts."""
    return json_safe(
        build_player_scouting_report(
            recommendation_data=resources.recommendation_data,
            performance_df=resources.short_term_data,
            long_term_df=resources.long_term_data,
            short_term_models=resources.short_term_models,
            long_term_models=resources.long_term_models,
            player_id=request.player_id,
            player_name=request.player_name,
            season=request.season,
            anchor_season=request.anchor_season,
            as_of_date=request.as_of_date,
            short_term_tasks=request.short_term_tasks,
            long_term_tasks=request.long_term_tasks,
            long_term_horizons=request.long_term_horizons,
            include_forecasts=request.include_forecasts,
        )
    )
