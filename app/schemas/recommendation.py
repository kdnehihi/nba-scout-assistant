from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    player_name: str
    season: str | None = None
    top_n: int = Field(default=5, ge=1, le=25)
    preset: str = "playing_profile"
    same_season: bool = True
    same_position_group: bool = True
    minutes_min: float | None = 500
