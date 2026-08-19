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


def _build_role_base(
    players: pd.DataFrame,
    season_stats: pd.DataFrame,
    game_logs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # Prefer game-log-derived production rates when game logs are available.
    """Return the base player-season role table before cleaning and dimension scoring."""
    role = season_stats.copy()
    if game_logs is None or game_logs.empty:
        return role

    from .features_long_term import build_player_season_summary

    production = build_player_season_summary(game_logs, players, season_stats).rename(
        columns={
            "pts_per_100": "points_per_100",
            "ast_per_100": "assists_per_100",
            "reb_per_100": "rebounds_per_100",
        }
    )
    production_cols = [
        column
        for column in [
            "player_id",
            "season",
            "points_per_100",
            "assists_per_100",
            "rebounds_per_100",
            "age_at_anchor",
        ]
        if column in production.columns
    ]
    role = role.merge(
        production[production_cols].drop_duplicates(["player_id", "season"]),
        on=["player_id", "season"],
        how="left",
        suffixes=("", "_game_logs"),
    )
    for column in ["points_per_100", "assists_per_100", "rebounds_per_100"]:
        game_log_column = f"{column}_game_logs"
        if game_log_column in role.columns:
            role[column] = role[game_log_column].combine_first(role.get(column))
            role = role.drop(columns=[game_log_column])
    if "age" not in role.columns and "age_at_anchor" in role.columns:
        role["age"] = role["age_at_anchor"]
    role = role.drop(columns=["age_at_anchor"], errors="ignore")
    return role


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
        - result["turnover_rate"].clip(0, 1) * 5 * 0.30
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


def build_role_features(
    players: pd.DataFrame,
    season_stats: pd.DataFrame,
    game_logs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # Join player metadata and season stats into role features.
    """Build player-season role features for similarity and forecasting."""
    role = _build_role_base(players, season_stats, game_logs)
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
