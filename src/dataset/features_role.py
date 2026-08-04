from __future__ import annotations

import pandas as pd

from .cleaning import add_missing_flags, fill_categorical_unknown, fill_numeric_median, percent_to_ratio


ROLE_NUMERIC_FEATURES = [
    "age",
    "height",
    "weight",
    "minutes",
    "usage_pct",
    "points_per_100",
    "assists_per_100",
    "rebounds_per_100",
    "true_shooting_pct",
    "three_point_attempt_rate",
    "free_throw_rate",
    "turnover_rate",
    "steal_rate",
    "block_rate",
    "defensive_rebound_rate",
    "foul_rate",
    "pace",
    "possessions",
    "offensive_rating",
    "defensive_rating",
]


def add_role_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    # Transform rate stats into interpretable scouting role dimensions.
    """Create interpretable role dimensions from player-season rate features."""
    result = df.copy()
    result["scoring_creation"] = (
        result["points_per_100"] * 0.45
        + result["usage_pct"] * 40 * 0.35
        + result["true_shooting_pct"] * 20 * 0.20
    )
    result["playmaking"] = (
        result["assists_per_100"] * 0.70
        + (1 - result["turnover_rate"].clip(0, 1)) * 5 * 0.30
    )
    result["shooting"] = (
        result["true_shooting_pct"] * 0.65
        + result["three_point_attempt_rate"].clip(0, 1) * 0.35
    )
    result["rim_pressure"] = result["free_throw_rate"] * 0.50 + result["usage_pct"] * 0.50
    result["rebounding"] = result["rebounds_per_100"] * 0.55 + result["defensive_rebound_rate"] * 20 * 0.45
    result["perimeter_defense"] = result["steal_rate"] * 0.75 + (1 - result["foul_rate"].clip(0, 1)) * 0.25
    result["interior_defense"] = (
        result["block_rate"] * 0.45
        + result["defensive_rebound_rate"] * 0.35
        + (1 - result["foul_rate"].clip(0, 1)) * 0.20
    )
    result["two_way_impact"] = result["offensive_rating"] - result["defensive_rating"]
    return result


def build_role_features(players: pd.DataFrame, season_stats: pd.DataFrame) -> pd.DataFrame:
    # Join player metadata and season stats into role features.
    """Build player-season role features for similarity and salary joins."""
    role = season_stats.copy()
    for column in [
        "usage_pct",
        "true_shooting_pct",
        "three_point_attempt_rate",
        "free_throw_rate",
        "turnover_rate",
        "steal_rate",
        "block_rate",
        "defensive_rebound_rate",
        "foul_rate",
    ]:
        if column in role.columns:
            role[column] = percent_to_ratio(role[column])

    player_cols = [column for column in ["player_id", "birth_date", "position", "height", "weight"] if column in players.columns]
    if player_cols:
        role = role.merge(
            players[player_cols].drop_duplicates("player_id"),
            on="player_id",
            how="left",
            suffixes=("", "_player"),
        )
        if "position_player" in role.columns:
            role["position"] = role.get("position").fillna(role["position_player"])
            role = role.drop(columns=["position_player"])
        for column in ["height", "weight"]:
            player_col = f"{column}_player"
            if player_col in role.columns:
                role[column] = role.get(column).fillna(role[player_col])
                role = role.drop(columns=[player_col])

    expected_numeric = [column for column in ROLE_NUMERIC_FEATURES if column in role.columns]
    role = add_missing_flags(role, expected_numeric + ["position"])
    role = fill_numeric_median(role, expected_numeric)
    role = fill_categorical_unknown(role, ["position", "team_id"])
    role = add_role_dimensions(role)

    role = role.drop_duplicates(["player_id", "season", "team_id"]).reset_index(drop=True)
    return role

