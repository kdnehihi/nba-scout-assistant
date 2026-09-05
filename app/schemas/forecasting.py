from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.common import PlayerLookupRequest

ShortTermTask = Literal["points", "assists", "rebounds"]
LongTermHorizon = Literal[1, 2, 3]
LongTermTask = Literal[
    "active_probability",
    "pts_per_36",
    "ast_per_36",
    "reb_per_36",
]


class PredictShortTermRequest(PlayerLookupRequest):
    season: str | None = None
    as_of_date: str | None = None
    tasks: list[ShortTermTask] = Field(
        default_factory=lambda: ["points", "assists", "rebounds"]
    )


class PredictLongTermRequest(PlayerLookupRequest):
    anchor_season: str | None = None
    tasks: list[LongTermTask] = Field(
        default_factory=lambda: [
            "active_probability",
            "pts_per_36",
            "ast_per_36",
            "reb_per_36",
        ]
    )
    horizons: list[LongTermHorizon] = Field(default_factory=lambda: [1, 2, 3])
