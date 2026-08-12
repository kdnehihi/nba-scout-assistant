from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.api_utils import json_safe
from src.pipelines.artifacts import (
    load_long_term_model_artifacts,
    load_short_term_model_artifacts,
)
from src.pipelines.forecasting import (
    load_long_term_prediction_data,
    load_short_term_prediction_data,
    predict_long_term_tasks,
    predict_short_term_tasks,
)
from src.pipelines.recommendation import (
    evaluate_recommendation_result,
    load_recommendation_pipeline_data,
    recommend_similar_players,
)
from src.pipelines.reporting import build_player_scouting_report, build_service_metadata


ShortTermTask = Literal["points", "assists", "rebounds"]
LongTermHorizon = Literal[1, 2, 3]
LongTermTask = Literal[
    "active_probability",
    "pts_per_36",
    "ast_per_36",
    "reb_per_36",
    "pts_per_100",
    "ast_per_100",
    "reb_per_100",
]


app = FastAPI(title="NBA Scout Assistant")


class PlayerLookupRequest(BaseModel):
    player_id: int | str | None = None
    player_name: str | None = None

    @model_validator(mode="after")
    def require_player_identifier(self):
        if self.player_id is None and self.player_name is None:
            raise ValueError("player_id or player_name is required")
        return self


class RecommendationRequest(BaseModel):
    player_name: str
    season: str | None = None
    top_n: int = Field(default=5, ge=1, le=25)
    preset: str = "playing_profile"
    same_season: bool = True
    same_position_group: bool = True
    minutes_min: float | None = 500


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


@app.on_event("startup")
def startup() -> None:
    app.state.recommendation_data = load_recommendation_pipeline_data("data")
    app.state.short_term_data = load_short_term_prediction_data("data")
    app.state.long_term_data = load_long_term_prediction_data("data")
    app.state.short_term_models = load_short_term_model_artifacts("artifacts")
    app.state.long_term_models = load_long_term_model_artifacts("artifacts")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return json_safe(
        build_service_metadata(
            recommendation_data=app.state.recommendation_data,
            performance_df=app.state.short_term_data,
            long_term_df=app.state.long_term_data,
            short_term_models=app.state.short_term_models,
            long_term_models=app.state.long_term_models,
        )
    )


@app.post("/recommendations")
def recommendations(request: RecommendationRequest):
    try:
        recs = recommend_similar_players(
            pipeline_data=app.state.recommendation_data,
            player_name=request.player_name or str(request.player_id),
            season=request.season,
            top_n=request.top_n,
            preset=request.preset,
            same_season=request.same_season,
            same_position_group=request.same_position_group,
            minutes_min=request.minutes_min,
        )
        diagnostics = evaluate_recommendation_result(
            pipeline_data=app.state.recommendation_data,
            recommendations=recs,
            top_n=request.top_n,
        )
        return json_safe(
            {
                "recommendations": recs,
                "diagnostics": diagnostics,
            }
        )
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/forecasts/short-term")
def short_term_forecast(request: PredictShortTermRequest):
    try:
        predictions = predict_short_term_tasks(
            performance_df=app.state.short_term_data,
            artifacts=app.state.short_term_models,
            tasks=request.tasks,
            player_id=request.player_id,
            player_name=request.player_name,
            season=request.season,
            as_of_date=request.as_of_date,
        )
        return json_safe({"predictions": predictions})
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/forecasts/long-term")
def long_term_forecast(request: PredictLongTermRequest):
    try:
        predictions = predict_long_term_tasks(
            long_term_df=app.state.long_term_data,
            artifacts=app.state.long_term_models,
            tasks=request.tasks,
            horizons=request.horizons,
            player_id=request.player_id,
            player_name=request.player_name,
            anchor_season=request.anchor_season,
        )
        return json_safe({"predictions": predictions})
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/players/scouting-report")
def scouting_report(request: ScoutingReportRequest):
    try:
        return json_safe(
            build_player_scouting_report(
                recommendation_data=app.state.recommendation_data,
                performance_df=app.state.short_term_data,
                long_term_df=app.state.long_term_data,
                short_term_models=app.state.short_term_models,
                long_term_models=app.state.long_term_models,
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
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
