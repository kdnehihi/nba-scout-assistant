from __future__ import annotations

from typing import Any

import pandas as pd

from src.api_utils import json_safe
from src.dataset.cleaning import normalize_name_key
from src.pipelines.forecasting import predict_long_term_tasks, predict_short_term_tasks
from src.pipelines.recommendation import (
    RecommendationPipelineData,
    build_recommended_player_detail,
)


def _json_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    # Convert dataframe slices to JSON-compatible records.
    """Return dataframe records with missing values converted to None."""
    records_df = df.head(limit).copy() if limit is not None else df.copy()
    return json_safe(records_df)


def _filter_player_rows(
    df: pd.DataFrame,
    player_id: int | str | None = None,
    player_name: str | None = None,
) -> pd.DataFrame:
    # Select player rows by ID when available, otherwise by normalized player name.
    """Return rows for one player request."""
    if df.empty:
        return df.copy()
    if player_id is not None and "player_id" in df.columns:
        return df[df["player_id"].astype(str).eq(str(player_id))].copy()
    if player_name is not None and "player_name" in df.columns:
        player_key = normalize_name_key(player_name)
        name_keys = (
            df["player_name_key"]
            if "player_name_key" in df.columns
            else df["player_name"].map(normalize_name_key)
        )
        mask = name_keys.eq(player_key)
        if not mask.any() and player_key:
            mask = name_keys.fillna("").str.contains(player_key, regex=False)
        return df[mask].copy()
    raise ValueError("player_id or player_name is required.")


def _filter_season(df: pd.DataFrame, season: str | None = None, season_col: str = "season") -> pd.DataFrame:
    # Narrow rows to a requested season when a season column exists.
    """Return rows filtered to a season when requested."""
    if season is None or season_col not in df.columns:
        return df.copy()
    return df[df[season_col].eq(season)].copy()


def latest_player_rows(
    df: pd.DataFrame,
    player_id: int | str | None = None,
    player_name: str | None = None,
    season: str | None = None,
    season_col: str = "season",
    sort_cols: list[str] | None = None,
    limit: int | None = 1,
) -> list[dict[str, Any]]:
    # Select latest player rows for report sections.
    """Return latest records for one player and optional season."""
    rows = _filter_player_rows(df, player_id=player_id, player_name=player_name)
    rows = _filter_season(rows, season=season, season_col=season_col)
    if rows.empty:
        return []
    ordered = rows.sort_values([col for col in (sort_cols or [season_col]) if col in rows.columns])
    if limit is not None:
        ordered = ordered.tail(limit)
    return _json_records(ordered)


def build_player_scouting_report(
    recommendation_data: RecommendationPipelineData,
    performance_df: pd.DataFrame,
    long_term_df: pd.DataFrame,
    short_term_models: dict[str, Any] | None = None,
    long_term_models: dict[tuple[str, int], Any] | None = None,
    player_id: int | str | None = None,
    player_name: str | None = None,
    season: str | None = None,
    anchor_season: str | None = None,
    as_of_date: str | None = None,
    short_term_tasks: list[str] | None = None,
    long_term_tasks: list[str] | None = None,
    long_term_horizons: list[int] | None = None,
    include_forecasts: bool = True,
) -> dict[str, Any]:
    # Build one player detail report for API display.
    """Return player context, compensation, signals, and optional forecasts."""
    player_rows = latest_player_rows(
        recommendation_data.recommendation_base,
        player_id=player_id,
        player_name=player_name,
        season=season,
        sort_cols=["season_start_year", "minutes"],
        limit=1,
    )
    if not player_rows:
        raise ValueError("No player profile rows matched the request.")

    selected_player_id = player_rows[0].get("player_id", player_id)
    selected_player_name = player_rows[0].get("player_name", player_name)

    compensation = build_recommended_player_detail(
        recommendation_data,
        player_id=selected_player_id,
        player_name=selected_player_name,
    )
    report: dict[str, Any] = {
        "player": player_rows[0],
        "compensation": {
            "latest_salary": compensation["latest_salary"],
            "salary_history": _json_records(compensation["salary_history"]),
            "contract_history": _json_records(compensation["contract_history"]),
        },
        "recent_performance": latest_player_rows(
            performance_df,
            player_id=selected_player_id,
            season=season,
            sort_cols=["as_of_date", "game_id"],
            limit=5,
        ),
        "long_term_anchor": latest_player_rows(
            long_term_df,
            player_id=selected_player_id,
            season=anchor_season,
            season_col="anchor_season",
            sort_cols=["anchor_season_start_year"],
            limit=1,
        ),
        "warnings": [],
    }

    if include_forecasts and short_term_models is not None:
        report["short_term_forecast"] = {}
        selected_short_tasks = list(short_term_tasks or short_term_models)
        for task in selected_short_tasks:
            if task not in short_term_models:
                report["warnings"].append(f"Short-term forecast unavailable for {task}: model not loaded")
                continue
            try:
                report["short_term_forecast"].update(
                    predict_short_term_tasks(
                        performance_df=performance_df,
                        artifacts=short_term_models,
                        tasks=[task],
                        player_id=selected_player_id,
                        season=season,
                        as_of_date=as_of_date,
                    )
                )
            except (KeyError, ValueError) as error:
                report["warnings"].append(f"Short-term forecast unavailable for {task}: {error}")

    if include_forecasts and long_term_models is not None:
        report["long_term_forecast"] = {}
        selected_long_tasks = list(long_term_tasks or sorted({task for task, _ in long_term_models}))
        for task in selected_long_tasks:
            task_artifacts = {
                key: artifact
                for key, artifact in long_term_models.items()
                if key[0] == task and (long_term_horizons is None or key[1] in long_term_horizons)
            }
            if not task_artifacts:
                report["warnings"].append(f"Long-term forecast unavailable for {task}: model not loaded")
                continue
            try:
                report["long_term_forecast"].update(
                    predict_long_term_tasks(
                        long_term_df=long_term_df,
                        artifacts=task_artifacts,
                        tasks=[task],
                        horizons=long_term_horizons,
                        player_id=selected_player_id,
                        anchor_season=anchor_season,
                    )
                )
            except (KeyError, ValueError) as error:
                report["warnings"].append(f"Long-term forecast unavailable for {task}: {error}")

    return report


def build_service_metadata(
    recommendation_data: RecommendationPipelineData,
    performance_df: pd.DataFrame,
    long_term_df: pd.DataFrame,
    short_term_models: dict[str, Any],
    long_term_models: dict[tuple[str, int], Any],
) -> dict[str, Any]:
    # Summarize loaded service data and model artifacts for local debugging.
    """Return loaded data/model metadata for the API."""
    return {
        "recommendation_base_rows": len(recommendation_data.recommendation_base),
        "performance_rows": len(performance_df),
        "long_term_rows": len(long_term_df),
        "salary_history_rows": len(recommendation_data.salary_history),
        "contract_history_rows": len(recommendation_data.contract_history),
        "short_term_models": sorted(short_term_models.keys()),
        "long_term_models": [
            {"task": task, "horizon": horizon}
            for task, horizon in sorted(long_term_models.keys())
        ],
        "recommendation_ranker": {
            "algorithm": recommendation_data.ranker_artifact.algorithm,
            "version": recommendation_data.ranker_artifact.version,
        }
        if recommendation_data.ranker_artifact is not None
        else {
            "algorithm": "season_normalized_euclidean",
            "version": "deterministic-v2",
        },
        "recommendation_seasons": sorted(recommendation_data.recommendation_base["season"].dropna().unique().tolist())
        if "season" in recommendation_data.recommendation_base.columns
        else [],
        "long_term_anchor_seasons": sorted(long_term_df["anchor_season"].dropna().unique().tolist())
        if "anchor_season" in long_term_df.columns
        else [],
    }
