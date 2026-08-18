from __future__ import annotations

from pydantic import Field

from app.schemas.common import PlayerLookupRequest
from app.schemas.forecasting import LongTermHorizon, LongTermTask, ShortTermTask


class ScoutingReportRequest(PlayerLookupRequest):
    season: str | None = None
    anchor_season: str | None = None
    as_of_date: str | None = None
    include_forecasts: bool = True
    short_term_tasks: list[ShortTermTask] = Field(
        default_factory=lambda: ["points", "assists", "rebounds"]
    )
    long_term_tasks: list[LongTermTask] = Field(
        default_factory=lambda: [
            "active_probability",
            "pts_per_36",
            "ast_per_36",
            "reb_per_36",
        ]
    )
    long_term_horizons: list[LongTermHorizon] = Field(default_factory=lambda: [1, 2, 3])
