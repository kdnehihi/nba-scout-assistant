from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import safe_per_36, season_label_to_start_year
from .splits import assign_long_term_temporal_split


NBA_REGULAR_SEASON_GAMES = 82
LONG_TERM_HORIZONS = [1, 2, 3]
LONG_TERM_CAREER_LAGS = 4
LONG_TERM_RECENT_GAMES = 20


def safe_per_100(numerator: pd.Series, possessions: pd.Series) -> pd.Series:
    # Scale counting stats to per-100-possession production rates.
    # Example: 25 points in 80 possessions -> 31.25 points per 100 possessions.
    """Convert counting stats to per-100-possession rates."""
    possessions = pd.to_numeric(possessions, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    return np.where(possessions.gt(0), numerator / possessions * 100, np.nan) #gt = greater than


def slope_last_values(values: pd.Series, min_points: int = 2) -> float:
    # Estimate a simple linear trend over ordered non-missing values.
    # Example: [10, 12, 14] -> positive slope; [14, 12, 10] -> negative slope.
    """Return a simple linear slope over ordered non-missing numeric values."""
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    if len(clean) < min_points:
        return np.nan
    x = np.arange(len(clean), dtype="float64")
    return float(np.polyfit(x, clean, 1)[0])


def build_player_season_summary(
    game_logs: pd.DataFrame,
    players: pd.DataFrame,
    season_stats: pd.DataFrame,
) -> pd.DataFrame:
    # Aggregate player game logs into one row per player-season.
    """Build player-season summaries used as long-term anchors and future targets."""
    if game_logs.empty:
        return pd.DataFrame()

    logs = game_logs.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    numeric_cols = [
        "minutes",
        "points",
        "assists",
        "rebounds",
        "fga",
        "fta",
        "fg3a",
        "true_shooting_pct",
        "rest_days",
    ]
    for column in numeric_cols:
        if column in logs.columns:
            logs[column] = pd.to_numeric(logs[column], errors="coerce")

    aggregation = {
        "player_name": ("player_name", "last"),
        "team_id": ("team_id", "last"),
        "season_end_date": ("game_date", "max"),
        "games_played": ("game_id", "nunique"),
        "total_minutes": ("minutes", "sum"),
        "minutes_per_game": ("minutes", "mean"),
        "points": ("points", "sum"),
        "assists": ("assists", "sum"),
        "rebounds": ("rebounds", "sum"),
    }
    for column in ["fga", "fta", "fg3a"]:
        if column in logs.columns:
            aggregation[column] = (column, "sum")
    for column in ["true_shooting_pct", "rest_days"]:
        if column in logs.columns:
            aggregation[f"{column}_avg" if column == "rest_days" else column] = (column, "mean")

    summary = logs.groupby(["player_id", "season"], sort=False).agg(**aggregation).reset_index()
    summary["season_start_year"] = summary["season"].map(season_label_to_start_year).astype("Int64")
    summary["availability_rate"] = (summary["games_played"] / NBA_REGULAR_SEASON_GAMES).clip(0, 1)
    summary["pts_per_36"] = safe_per_36(summary["points"], summary["total_minutes"])
    summary["ast_per_36"] = safe_per_36(summary["assists"], summary["total_minutes"])
    summary["reb_per_36"] = safe_per_36(summary["rebounds"], summary["total_minutes"])
    for source, output in [("fga", "fga_per_36"), ("fta", "fta_per_36"), ("fg3a", "fg3a_per_36")]:
        if source in summary.columns:
            summary[output] = safe_per_36(summary[source], summary["total_minutes"])

    if not season_stats.empty:
        advanced_cols = [
            column
            for column in ["player_id", "season", "usage_pct", "pace", "offensive_rating", "defensive_rating"]
            if column in season_stats.columns
        ]
        if {"player_id", "season"}.issubset(advanced_cols):
            summary = summary.merge(
                season_stats[advanced_cols].drop_duplicates(["player_id", "season"]),
                on=["player_id", "season"],
                how="left",
            )

    if "pace" in summary.columns:
        season_pace = summary.groupby("season")["pace"].transform("median")
        pace_by_year = (
            summary[["season_start_year", "season", "pace"]]
            .drop_duplicates()
            .groupby("season_start_year")["pace"]
            .median()
            .sort_index()
            .ffill()
            .bfill()
        )
        fallback_pace_by_year = summary["season_start_year"].map(pace_by_year)
        summary["pace_for_possessions"] = (
            pd.to_numeric(summary["pace"], errors="coerce").fillna(season_pace).fillna(fallback_pace_by_year)
        )
    else:
        summary["pace_for_possessions"] = np.nan

    summary["estimated_possessions"] = (
        summary["pace_for_possessions"] * pd.to_numeric(summary["total_minutes"], errors="coerce") / 48
    )
    summary["pts_per_100"] = safe_per_100(summary["points"], summary["estimated_possessions"])
    summary["ast_per_100"] = safe_per_100(summary["assists"], summary["estimated_possessions"])
    summary["reb_per_100"] = safe_per_100(summary["rebounds"], summary["estimated_possessions"])

    player_cols = [column for column in ["player_id", "birth_date", "position", "height", "weight", "age"] if column in players.columns]
    if player_cols:
        summary = summary.merge(players[player_cols].drop_duplicates("player_id"), on="player_id", how="left")

    if "birth_date" in summary.columns:
        birth_date = pd.to_datetime(summary["birth_date"], errors="coerce")
        summary["age_at_anchor"] = (summary["season_end_date"] - birth_date).dt.days / 365.25
    elif "age" in summary.columns:
        summary["age_at_anchor"] = pd.to_numeric(summary["age"], errors="coerce")
    else:
        summary["age_at_anchor"] = np.nan

    summary = summary.sort_values(["player_id", "season_start_year"]).reset_index(drop=True)
    summary["career_games"] = summary.groupby("player_id")["games_played"].cumsum()
    summary["career_minutes"] = summary.groupby("player_id")["total_minutes"].cumsum()
    summary["years_in_league"] = summary.groupby("player_id").cumcount() + 1
    return summary


def build_recent_game_anchor_features(
    game_logs: pd.DataFrame,
    recent_games: int = LONG_TERM_RECENT_GAMES,
) -> pd.DataFrame:
    # Summarize the player's most recent games within each anchor season.
    """Build recent-form features from the last N games of each player-season."""
    if game_logs.empty:
        return pd.DataFrame()

    logs = game_logs.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    for column in ["minutes", "points", "assists", "rebounds", "fga", "fta", "true_shooting_pct", "rest_days", "pace"]:
        if column in logs.columns:
            logs[column] = pd.to_numeric(logs[column], errors="coerce")

    recent_rows: list[dict[str, object]] = []
    for (player_id, season), group in logs.sort_values("game_date").groupby(["player_id", "season"], sort=False):
        recent = group.tail(recent_games).copy()
        total_minutes = pd.to_numeric(recent.get("minutes"), errors="coerce").sum()
        row: dict[str, object] = {
            "player_id": player_id,
            "season": season,
            "recent_games_count": len(recent),
            "recent_minutes_per_game": recent["minutes"].mean() if "minutes" in recent.columns else np.nan,
            "recent_true_shooting_pct": recent["true_shooting_pct"].mean() if "true_shooting_pct" in recent.columns else np.nan,
            "recent_rest_days_avg": recent["rest_days"].mean() if "rest_days" in recent.columns else np.nan,
            "recent_minutes_trend": slope_last_values(recent["minutes"]) if "minutes" in recent.columns else np.nan,
            "recent_pts_trend": slope_last_values(recent["points"]) if "points" in recent.columns else np.nan,
            "recent_ast_trend": slope_last_values(recent["assists"]) if "assists" in recent.columns else np.nan,
            "recent_reb_trend": slope_last_values(recent["rebounds"]) if "rebounds" in recent.columns else np.nan,
        }
        for stat, output in [
            ("points", "recent_pts_per_36"),
            ("assists", "recent_ast_per_36"),
            ("rebounds", "recent_reb_per_36"),
            ("fga", "recent_fga_per_36"),
            ("fta", "recent_fta_per_36"),
        ]:
            if stat in recent.columns:
                row[output] = float(recent[stat].sum() / total_minutes * 36) if total_minutes > 0 else np.nan
        recent_rows.append(row)

    return pd.DataFrame(recent_rows)


def add_lagged_season_features(
    anchor_df: pd.DataFrame,
    career_lags: int = LONG_TERM_CAREER_LAGS,
) -> pd.DataFrame:
    # Flatten current and prior player seasons into one anchor feature row.
    """Add lagged player-season features and recent 3-year trend slopes."""
    feature_cols = [
        "age_at_anchor",
        "games_played",
        "minutes_per_game",
        "total_minutes",
        "availability_rate",
        "pts_per_36",
        "ast_per_36",
        "reb_per_36",
        "pts_per_100",
        "ast_per_100",
        "reb_per_100",
        "fga_per_36",
        "fta_per_36",
        "true_shooting_pct",
        "usage_pct",
    ]
    feature_cols = [column for column in feature_cols if column in anchor_df.columns]
    rows: list[dict[str, object]] = []

    ordered = anchor_df.sort_values(["player_id", "season_start_year"]).copy()
    for _, group in ordered.groupby("player_id", sort=False):
        group = group.reset_index(drop=True)
        for idx, anchor in group.iterrows():
            row: dict[str, object] = {
                "player_id": anchor["player_id"],
                "player_name": anchor.get("player_name"),
                "anchor_season": anchor["season"],
                "anchor_season_start_year": anchor["season_start_year"],
                "anchor_date": anchor["season_end_date"],
                "team_id": anchor.get("team_id"),
                "position": anchor.get("position", "UNK"),
                "height": anchor.get("height"),
                "weight": anchor.get("weight"),
                "age_at_anchor": anchor.get("age_at_anchor"),
                "years_in_league": anchor.get("years_in_league"),
                "career_games": anchor.get("career_games"),
                "career_minutes": anchor.get("career_minutes"),
            }
            history = group.loc[:idx].tail(career_lags).reset_index(drop=True)
            for lag in range(career_lags):
                hist_idx = len(history) - 1 - lag
                for feature in feature_cols:
                    row[f"{feature}_lag_{lag}"] = history.loc[hist_idx, feature] if hist_idx >= 0 else np.nan

            trend_window = history.tail(3)
            for feature in [
                "games_played",
                "minutes_per_game",
                "pts_per_36",
                "ast_per_36",
                "reb_per_36",
                "pts_per_100",
                "ast_per_100",
                "reb_per_100",
                "total_minutes",
            ]:
                if feature in trend_window.columns:
                    row[f"{feature}_slope_3yr"] = slope_last_values(trend_window[feature])
            rows.append(row)

    return pd.DataFrame(rows)


def add_future_horizon_targets(
    anchor_features: pd.DataFrame,
    season_summary: pd.DataFrame,
    horizons: list[int] = LONG_TERM_HORIZONS,
) -> pd.DataFrame:
    # Attach observed future-season labels for each forecast horizon.
    """Add h1/h2/h3 active, games, per-minute, per-possession, and per-game targets."""
    result = anchor_features.copy()
    future_lookup = season_summary.set_index(["player_id", "season_start_year"])
    max_observed_season_start = int(season_summary["season_start_year"].max())

    def empty_target(horizon: int, active_value: float | int | None, games_value: float | int | None) -> dict[str, object]:
        return {
            f"active_h{horizon}": active_value,
            f"games_played_h{horizon}": games_value,
            f"minutes_per_game_h{horizon}": np.nan,
            f"pts_per_36_h{horizon}": np.nan,
            f"ast_per_36_h{horizon}": np.nan,
            f"reb_per_36_h{horizon}": np.nan,
            f"pts_per_100_h{horizon}": np.nan,
            f"ast_per_100_h{horizon}": np.nan,
            f"reb_per_100_h{horizon}": np.nan,
            f"pts_per_game_h{horizon}": np.nan,
            f"ast_per_game_h{horizon}": np.nan,
            f"reb_per_game_h{horizon}": np.nan,
        }

    for horizon in horizons:
        target_rows: list[dict[str, object]] = []
        for _, row in result[["player_id", "anchor_season_start_year"]].iterrows():
            future_season_start = row["anchor_season_start_year"] + horizon
            key = (row["player_id"], future_season_start)
            if key in future_lookup.index:
                target = future_lookup.loc[key]
                if isinstance(target, pd.DataFrame):
                    target = target.iloc[0]
                active = int(pd.notna(target.get("games_played")) and target.get("games_played", 0) > 0)
                minutes_per_game = target.get("minutes_per_game")
                pts_per_36 = target.get("pts_per_36")
                ast_per_36 = target.get("ast_per_36")
                reb_per_36 = target.get("reb_per_36")
                target_rows.append(
                    {
                        f"active_h{horizon}": active,
                        f"games_played_h{horizon}": target.get("games_played"),
                        f"minutes_per_game_h{horizon}": minutes_per_game,
                        f"pts_per_36_h{horizon}": pts_per_36,
                        f"ast_per_36_h{horizon}": ast_per_36,
                        f"reb_per_36_h{horizon}": reb_per_36,
                        f"pts_per_100_h{horizon}": target.get("pts_per_100"),
                        f"ast_per_100_h{horizon}": target.get("ast_per_100"),
                        f"reb_per_100_h{horizon}": target.get("reb_per_100"),
                        f"pts_per_game_h{horizon}": minutes_per_game * pts_per_36 / 36 if pd.notna(minutes_per_game) and pd.notna(pts_per_36) else np.nan,
                        f"ast_per_game_h{horizon}": minutes_per_game * ast_per_36 / 36 if pd.notna(minutes_per_game) and pd.notna(ast_per_36) else np.nan,
                        f"reb_per_game_h{horizon}": minutes_per_game * reb_per_36 / 36 if pd.notna(minutes_per_game) and pd.notna(reb_per_36) else np.nan,
                    }
                )
            elif future_season_start <= max_observed_season_start:
                target_rows.append(empty_target(horizon, active_value=0, games_value=0))
            else:
                target_rows.append(empty_target(horizon, active_value=np.nan, games_value=np.nan))

        result = pd.concat([result.reset_index(drop=True), pd.DataFrame(target_rows)], axis=1)
        result[f"age_at_h{horizon}"] = result["age_at_anchor"] + horizon

    return result


def build_long_term_training(
    game_logs: pd.DataFrame,
    players: pd.DataFrame,
    season_stats: pd.DataFrame,
) -> pd.DataFrame:
    # Create the final season-anchor table for long-term forecasting.
    """Build the notebook-consistent long-term player forecast training dataset."""
    season_summary = build_player_season_summary(game_logs, players, season_stats)
    recent_features = build_recent_game_anchor_features(game_logs)
    anchor_features = add_lagged_season_features(season_summary)

    if not recent_features.empty:
        anchor_features = (
            anchor_features.merge(
                recent_features,
                left_on=["player_id", "anchor_season"],
                right_on=["player_id", "season"],
                how="left",
            )
            .drop(columns=["season"], errors="ignore")
        )

    long_term = add_future_horizon_targets(anchor_features, season_summary)
    required_future_cols = [f"active_h{horizon}" for horizon in LONG_TERM_HORIZONS]
    long_term = long_term.dropna(subset=required_future_cols).copy()
    long_term["split"] = long_term["anchor_season"].map(assign_long_term_temporal_split)
    long_term = long_term[long_term["split"].isin(["train", "validation", "test"])].copy()
    return long_term.sort_values(["anchor_season_start_year", "player_id"]).reset_index(drop=True)
