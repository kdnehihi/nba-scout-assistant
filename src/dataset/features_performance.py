from __future__ import annotations

import numpy as np
import pandas as pd

from .splits import assign_temporal_split


STAT_MAP = {
    "pts": "points",
    "ast": "assists",
    "reb": "rebounds",
    "min": "minutes",
}


def _column(df: pd.DataFrame, canonical: str, fallback: str) -> str:
    # Resolve canonical or fallback stat column names.
    if canonical in df.columns:
        return canonical
    if fallback in df.columns:
        return fallback
    raise KeyError(f"Missing required column: {canonical} or {fallback}")


def future_average(values: pd.Series, horizon: int = 5) -> pd.Series:
    # Build next-horizon average targets without including the current row.
    """Return future average where row t uses t+1 through t+horizon."""
    future_values = [values.shift(-step) for step in range(1, horizon + 1)]
    return sum(future_values) / horizon


def add_rolling_player_features(game_logs: pd.DataFrame, min_history: int = 5) -> pd.DataFrame:
    # Create rolling form, season average, delta, and future target features.
    """Create point-in-time rolling form features and next-five-game targets."""
    logs = game_logs.copy()
    logs["as_of_date"] = pd.to_datetime(logs.get("as_of_date", logs["game_date"]))
    logs = logs.sort_values(["player_id", "season", "as_of_date", "game_id"]).reset_index(drop=True)

    point_col = _column(logs, "points", "pts")
    assist_col = _column(logs, "assists", "ast")
    rebound_col = _column(logs, "rebounds", "reb")
    minute_col = _column(logs, "minutes", "min")

    for column in [point_col, assist_col, rebound_col, minute_col]:
        logs[column] = pd.to_numeric(logs[column], errors="coerce")

    grouped = logs.groupby(["player_id", "season"], group_keys=False, sort=False)
    for prefix, source_col in [("pts", point_col), ("ast", assist_col), ("reb", rebound_col)]:
        logs[f"{prefix}_last_5"] = grouped[source_col].transform(lambda s: s.rolling(5, min_periods=min_history).mean())
        logs[f"{prefix}_last_10"] = grouped[source_col].transform(lambda s: s.rolling(10, min_periods=min_history).mean())
        logs[f"{prefix}_season_avg"] = grouped[source_col].transform(lambda s: s.expanding(min_periods=min_history).mean())
        logs[f"{prefix}_last_5_minus_season_avg"] = logs[f"{prefix}_last_5"] - logs[f"{prefix}_season_avg"]
        logs[f"{prefix}_last_10_minus_season_avg"] = logs[f"{prefix}_last_10"] - logs[f"{prefix}_season_avg"]
        logs[f"{prefix}_last_5_minus_last_10"] = logs[f"{prefix}_last_5"] - logs[f"{prefix}_last_10"]

    logs["min_last_5"] = grouped[minute_col].transform(lambda s: s.rolling(5, min_periods=min_history).mean())
    logs["min_last_10"] = grouped[minute_col].transform(lambda s: s.rolling(10, min_periods=min_history).mean())
    logs["min_season_avg"] = grouped[minute_col].transform(lambda s: s.expanding(min_periods=min_history).mean())
    logs["min_last_5_minus_season_avg"] = logs["min_last_5"] - logs["min_season_avg"]
    logs["min_last_10_minus_season_avg"] = logs["min_last_10"] - logs["min_season_avg"]
    logs["min_last_5_minus_last_10"] = logs["min_last_5"] - logs["min_last_10"]

    grouped = logs.groupby(["player_id", "season"], group_keys=False, sort=False)
    for prefix, source_col in [("pts", point_col), ("ast", assist_col), ("reb", rebound_col)]:
        logs[f"target_next_5_{prefix}_avg"] = grouped[source_col].transform(future_average)

    return logs.reset_index(drop=True)


def build_performance_training(game_logs: pd.DataFrame) -> pd.DataFrame:
    # Filter short-term rows to complete features and targets.
    """Build clean short-term performance training data."""
    features = add_rolling_player_features(game_logs)
    features["split"] = features["season"].map(assign_temporal_split)
    required = [
        "pts_last_5",
        "pts_last_10",
        "pts_season_avg",
        "ast_last_5",
        "ast_last_10",
        "ast_season_avg",
        "reb_last_5",
        "reb_last_10",
        "reb_season_avg",
        "min_last_5",
        "min_last_10",
        "min_season_avg",
        "target_next_5_pts_avg",
        "target_next_5_ast_avg",
        "target_next_5_reb_avg",
    ]
    existing_required = [column for column in required if column in features.columns]
    features = features.dropna(subset=existing_required).copy()
    features = features[features["split"].isin(["train", "validation", "test"])].copy()
    return features.reset_index(drop=True)
