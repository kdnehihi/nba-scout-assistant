from __future__ import annotations

import pandas as pd

from .cleaning import season_label_to_start_year


MIN_MODELED_UNIQUE_GAMES = 1000
MIN_MODELED_TEAMS = 30


def summarize_game_log_season_coverage(game_logs: pd.DataFrame) -> pd.DataFrame:
    # Summarize season-level coverage from canonical game logs.
    """Return one coverage row per season from game logs."""
    if game_logs.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "season_start_year",
                "rows",
                "unique_games",
                "players",
                "teams",
                "min_date",
                "max_date",
                "is_modeling_complete",
            ]
        )

    logs = game_logs.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")
    coverage = (
        logs.groupby("season", dropna=False)
        .agg(
            rows=("game_id", "size"),
            unique_games=("game_id", "nunique"),
            players=("player_id", "nunique"),
            teams=("team_id", "nunique"),
            min_date=("game_date", "min"),
            max_date=("game_date", "max"),
        )
        .reset_index()
    )
    coverage["season_start_year"] = coverage["season"].map(season_label_to_start_year).astype("Int64")
    coverage["is_modeling_complete"] = (
        coverage["unique_games"].ge(MIN_MODELED_UNIQUE_GAMES)
        & coverage["teams"].ge(MIN_MODELED_TEAMS)
    )
    return coverage.sort_values("season_start_year").reset_index(drop=True)


def latest_complete_modeling_season(game_logs: pd.DataFrame) -> str:
    # Find the latest season with enough game coverage for modeling.
    """Return the latest season considered complete enough for modeling."""
    coverage = summarize_game_log_season_coverage(game_logs)
    complete = coverage[coverage["is_modeling_complete"]].copy()
    if complete.empty:
        if coverage.empty:
            raise ValueError("No game-log season has enough coverage for modeling.")
        return str(coverage.sort_values("season_start_year").iloc[-1]["season"])
    latest = complete.sort_values("season_start_year").iloc[-1]
    return str(latest["season"])


def modeling_seasons_through_latest_complete(game_logs: pd.DataFrame) -> set[str]:
    # Keep seasons no later than the latest complete modeling season.
    """Return all seasons up to the latest complete modeling season."""
    coverage = summarize_game_log_season_coverage(game_logs)
    latest_season = latest_complete_modeling_season(game_logs)
    latest_year = season_label_to_start_year(latest_season)
    return set(
        coverage.loc[
            coverage["season_start_year"].le(latest_year),
            "season",
        ].dropna().astype(str)
    )


def filter_to_modeling_seasons(df: pd.DataFrame, seasons: set[str], season_col: str = "season") -> pd.DataFrame:
    # Drop rows after the latest complete modeling season.
    """Return rows whose season is in the approved modeling season set."""
    if df.empty or season_col not in df.columns:
        return df.copy()
    return df[df[season_col].astype(str).isin(seasons)].copy()
