from __future__ import annotations

from typing import Any

from app.schemas.recommendation import RecommendationRequest
from app.services.cache import (
    build_cache_key,
    cache_ttl_seconds,
    get_or_set_cached_json,
)
from app.services.resources import AppResources
from src.api_utils import json_safe
from src.pipelines.recommendation import (
    evaluate_recommendation_result,
    recommend_similar_players,
)


def build_recommendation_response(
    resources: AppResources,
    request: RecommendationRequest,
) -> dict[str, Any]:
    """Return recommendations plus deterministic diagnostics."""
    key = build_cache_key(
        "recommendations",
        resources.runtime_version,
        request.model_dump(mode="json"),
    )

    def build() -> dict[str, Any]:
        recommendations = recommend_similar_players(
            pipeline_data=resources.recommendation_data,
            player_name=request.player_name,
            season=request.season,
            top_n=request.top_n,
            preset=request.preset,
            same_season=request.same_season,
            same_position_group=request.same_position_group,
            minutes_min=request.minutes_min,
        )
        diagnostics = evaluate_recommendation_result(
            pipeline_data=resources.recommendation_data,
            recommendations=recommendations,
            top_n=request.top_n,
        )
        return json_safe(
            {
                "recommendations": recommendations,
                "diagnostics": diagnostics,
            }
        )

    return get_or_set_cached_json(
        resources.response_cache,
        key,
        cache_ttl_seconds("RECOMMENDATION_CACHE_TTL_SECONDS", 21600),
        build,
    )
