from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import safe_per_36, season_label_to_start_year
from .splits import assign_long_term_temporal_split


NBA_REGULAR_SEASON_GAMES = 82
LONG_TERM_HORIZONS = [1, 2, 3]
LONG_TERM_LAGS = 4


def build_player_season_summary(
    game_logs: pd.DataFrame,
    players: pd.DataFrame,
    season_stats: pd.DataFrame,
) -> pd.DataFrame:
    # Aggregate game logs into season-anchor player summaries.
    """Aggregate game logs to player-season summaries for long-term anchors."""
    logs = game_logs.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    for column in ["minutes", "points", "assists", "rebounds", "fga", "fta", "fg3a", "true_shooting_pct", "rest_days"]:
        if column in logs.columns:
            logs[column] = pd.to_numeric(logs[column], errors="coerce")

    grouped = logs.groupby(["player_id", "season"], sort=False)
    summary = grouped.agg(
        player_name=("player_name", "last"),
        team_id=("team_id", "last"),
        season_end_date=("game_date", "max"),
        games_played=("game_id", "nunique"),
        total_minutes=("minutes", "sum"),
        minutes_per_game=("minutes", "mean"),
        points=("points", "sum"),
        assists=("assists", "sum"),
        rebounds=("rebounds", "sum"),
    ).reset_index()
    summary["season_start_year"] = summary["season"].map(season_label_to_start_year).astype("Int64")
    summary["availability_rate"] = (summary["games_played"] / NBA_REGULAR_SEASON_GAMES).clip(0, 1)
    summary["pts_per_36"] = safe_per_36(summary["points"], summary["total_minutes"])
    summary["ast_per_36"] = safe_per_36(summary["assists"], summary["total_minutes"])
    summary["reb_per_36"] = safe_per_36(summary["rebounds"], summary["total_minutes"])

    if not season_stats.empty:
        advanced_cols = [column for column in ["player_id", "season", "usage_pct", "pace", "offensive_rating", "defensive_rating"] if column in season_stats.columns]
        if {"player_id", "season"}.issubset(advanced_cols):
            summary = summary.merge(
                season_stats[advanced_cols].drop_duplicates(["player_id", "season"]),
                on=["player_id", "season"],
                how="left",
            )

    player_cols = [column for column in ["player_id", "birth_date", "position", "height", "weight"] if column in players.columns]
    if player_cols:
        summary = summary.merge(players[player_cols].drop_duplicates("player_id"), on="player_id", how="left")
    if "birth_date" in summary.columns:
        birth_date = pd.to_datetime(summary["birth_date"], errors="coerce")
        summary["age_at_anchor"] = (summary["season_end_date"] - birth_date).dt.days / 365.25
    else:
        summary["age_at_anchor"] = np.nan

    summary = summary.sort_values(["player_id", "season_start_year"]).reset_index(drop=True)
    summary["career_games"] = summary.groupby("player_id")["games_played"].cumsum()
    summary["career_minutes"] = summary.groupby("player_id")["total_minutes"].cumsum()
    summary["years_in_league"] = summary.groupby("player_id").cumcount() + 1
    return summary


def add_lag_features(summary: pd.DataFrame) -> pd.DataFrame:
    # Attach prior-season player trajectory features to each anchor.
    """Add lagged player-season features for each anchor row."""
    result = summary.copy()
    lag_features = [
        "age_at_anchor",
        "games_played",
        "minutes_per_game",
        "total_minutes",
        "availability_rate",
        "pts_per_36",
        "ast_per_36",
        "reb_per_36",
        "usage_pct",
    ]
    for lag in range(LONG_TERM_LAGS):
        for feature in lag_features:
            if feature in result.columns:
                result[f"{feature}_lag_{lag}"] = result.groupby("player_id")[feature].shift(lag)
    return result


def add_future_targets(summary: pd.DataFrame) -> pd.DataFrame:
    # Attach future horizon labels used by long-term models.
    """Add h1/h2/h3 future availability and per-36 production targets."""
    result = summary.copy()
    for horizon in LONG_TERM_HORIZONS:
        for source, target in [
            ("games_played", f"games_played_h{horizon}"),
            ("pts_per_36", f"pts_per_36_h{horizon}"),
            ("ast_per_36", f"ast_per_36_h{horizon}"),
            ("reb_per_36", f"reb_per_36_h{horizon}"),
        ]:
            result[target] = result.groupby("player_id")[source].shift(-horizon)
        result[f"active_h{horizon}"] = result[f"games_played_h{horizon}"].gt(0).astype("Int64")
        result[f"low_availability_h{horizon}"] = result[f"games_played_h{horizon}"].le(20).astype("Int64")
        result[f"high_availability_h{horizon}"] = result[f"games_played_h{horizon}"].ge(61).astype("Int64")
    return result


def build_long_term_training(
    game_logs: pd.DataFrame,
    players: pd.DataFrame,
    season_stats: pd.DataFrame,
) -> pd.DataFrame:
    # Create the final long-term trajectory training table.
    """Build long-term player trajectory training data."""
    summary = build_player_season_summary(game_logs, players, season_stats)
    long_term = add_future_targets(add_lag_features(summary))
    required = [f"active_h{horizon}" for horizon in LONG_TERM_HORIZONS]
    long_term = long_term.dropna(subset=required).copy()
    long_term["anchor_season"] = long_term["season"]
    long_term["anchor_season_start_year"] = long_term["season_start_year"]
    long_term["anchor_date"] = long_term["season_end_date"]
    long_term["split"] = long_term["anchor_season"].map(assign_long_term_temporal_split)
    long_term = long_term[long_term["split"].isin(["train", "validation", "test"])].copy()
    return long_term.reset_index(drop=True)

