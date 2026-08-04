from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ConsistencyConfig, TrendConfig


def season_start_year(season: object) -> int:
    # Extract season start year from canonical NBA season labels.
    """Return the start year for a season label such as '2024-25'."""
    return int(str(season)[:4])


def classify_direction(delta: pd.Series, tolerance: float) -> pd.Series:
    # Convert numeric deltas into scout-readable direction labels.
    """Return improving/stable/declining labels for a numeric delta series."""
    return pd.Series(
        np.select([delta > tolerance, delta < -tolerance], ["improving", "declining"], default="stable"),
        index=delta.index,
    )


def robust_zscore(values: pd.Series) -> pd.Series:
    # Compute robust z-scores for volatility comparisons within a season.
    """Return median-absolute-deviation z-scores, falling back to standard z-scores."""
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if pd.isna(mad) or mad == 0:
        std = numeric.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(numeric)), index=numeric.index)
        return (numeric - numeric.mean()) / std
    return 0.6745 * (numeric - median) / mad


def build_player_trend_signals(
    performance_df: pd.DataFrame,
    config: TrendConfig = TrendConfig(),
) -> pd.DataFrame:
    # Build latest player-season trend labels from clean point-in-time performance features.
    """Return latest player-season trend signals for production and minutes."""
    latest = (
        performance_df.sort_values(["player_id", "season", "as_of_date", "game_id"])
        .groupby(["player_id", "season"], as_index=False)
        .tail(1)
        .copy()
    )
    latest["season_start_year"] = latest["season"].map(season_start_year)
    latest["pts_recent_delta"] = latest["pts_last_5"] - latest["pts_season_avg"]
    latest["ast_recent_delta"] = latest["ast_last_5"] - latest["ast_season_avg"]
    latest["reb_recent_delta"] = latest["reb_last_5"] - latest["reb_season_avg"]
    latest["min_recent_delta"] = latest["min_last_5"] - latest["min_season_avg"]
    latest["pts_trend"] = classify_direction(latest["pts_recent_delta"], config.points_tolerance)
    latest["ast_trend"] = classify_direction(latest["ast_recent_delta"], config.assists_tolerance)
    latest["reb_trend"] = classify_direction(latest["reb_recent_delta"], config.rebounds_tolerance)
    latest["minutes_trend"] = classify_direction(latest["min_recent_delta"], config.minutes_tolerance)

    score_map = {"improving": 1, "stable": 0, "declining": -1}
    latest["production_trend_score"] = (
        latest["pts_trend"].map(score_map)
        + latest["ast_trend"].map(score_map)
        + latest["reb_trend"].map(score_map)
    )
    latest["overall_trend"] = classify_direction(latest["production_trend_score"], config.overall_tolerance)
    output_cols = [
        "player_id",
        "season",
        "season_start_year",
        "team_id",
        "as_of_date",
        "pts_recent_delta",
        "ast_recent_delta",
        "reb_recent_delta",
        "min_recent_delta",
        "pts_trend",
        "ast_trend",
        "reb_trend",
        "minutes_trend",
        "production_trend_score",
        "overall_trend",
    ]
    return latest[output_cols].reset_index(drop=True)


def build_player_consistency_signals(
    performance_df: pd.DataFrame,
    config: ConsistencyConfig = ConsistencyConfig(),
    min_games: int | None = None,
) -> pd.DataFrame:
    # Summarize player-season production volatility from observed game logs.
    """Return player-season consistency and volatility descriptors."""
    min_required_games = config.min_games if min_games is None else min_games
    rows: list[dict[str, object]] = []
    for keys, group in performance_df.groupby(["player_id", "season", "team_id"], dropna=False):
        player_id, season, team_id = keys
        if len(group) < min_required_games:
            continue
        row: dict[str, object] = {
            "player_id": player_id,
            "season": season,
            "season_start_year": season_start_year(season),
            "team_id": team_id,
            "games_observed": len(group),
        }
        for stat in ["pts", "ast", "reb", "min"]:
            values = pd.to_numeric(group[stat], errors="coerce").dropna()
            if values.empty:
                continue
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=0))
            row[f"{stat}_mean"] = mean_value
            row[f"{stat}_std"] = std_value
            row[f"{stat}_cv"] = float(std_value / mean_value) if mean_value > 0 else np.nan
            row[f"{stat}_p20"] = float(values.quantile(config.lower_quantile))
            row[f"{stat}_p50"] = float(values.quantile(config.median_quantile))
            row[f"{stat}_p80"] = float(values.quantile(config.upper_quantile))
            row[f"{stat}_above_mean_rate"] = float((values > mean_value).mean())
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for stat in ["pts", "ast", "reb", "min"]:
        cv_col = f"{stat}_cv"
        if cv_col in result.columns:
            result[f"{stat}_volatility_z"] = result.groupby("season")[cv_col].transform(robust_zscore)
            result[f"{stat}_consistency_score"] = (-result[f"{stat}_volatility_z"]).clip(-3, 3)
    volatility_cols = [c for c in ["pts_volatility_z", "ast_volatility_z", "reb_volatility_z", "min_volatility_z"] if c in result.columns]
    result["overall_volatility_score"] = result[volatility_cols].mean(axis=1)
    result["consistency_label"] = pd.Series(
        np.select(
            [
                result["overall_volatility_score"] <= config.consistent_threshold,
                result["overall_volatility_score"] >= config.volatile_threshold,
            ],
            ["consistent", "volatile"],
            default="balanced",
        ),
        index=result.index,
    )
    return result.sort_values(["season_start_year", "player_id"]).reset_index(drop=True)
