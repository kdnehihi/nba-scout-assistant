from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RecommendationPreset = Literal[
    "playing_profile",
    "role_similarity",
    "scoring_profile",
    "defensive_profile",
    "workload_fit",
    "physical_role_fit",
]


class RecommendationRequest(BaseModel):
    player_name: str
    season: str | None = None
    top_n: int = Field(default=5, ge=1, le=25)
    preset: RecommendationPreset = "playing_profile"
    same_season: bool = True
    same_position_group: bool = True
    minutes_min: float | None = 500
