from __future__ import annotations

import pandas as pd

from evaluation.evaluate_ranges import evaluate_short_term_ranges

from .config import RangeConfig, STAT_TARGETS


STAT_CONFIG = {
    "pts": {"actual": "pts", "last_5": "pts_last_5", "last_10": "pts_last_10", "season_avg": "pts_season_avg"},
    "ast": {"actual": "ast", "last_5": "ast_last_5", "last_10": "ast_last_10", "season_avg": "ast_season_avg"},
    "reb": {"actual": "reb", "last_5": "reb_last_5", "last_10": "reb_last_10", "season_avg": "reb_season_avg"},
}


def add_recent_rolling_volatility(performance_df: pd.DataFrame, config: RangeConfig = RangeConfig()) -> pd.DataFrame:
    # Add rolling player-season standard deviations through row t.
    """Return performance rows with recent rolling volatility columns."""
    result = performance_df.sort_values(["player_id", "season", "as_of_date", "game_id"]).copy()
    for stat in STAT_CONFIG:
        result[f"{stat}_rolling_std_{config.rolling_std_window}"] = (
            result.groupby(["player_id", "season"])[stat]
            .transform(lambda s: s.rolling(config.rolling_std_window, min_periods=config.rolling_std_min_periods).std(ddof=0))
        )
    return result


def build_short_term_floor_ceiling_signals(
    performance_df: pd.DataFrame,
    config: RangeConfig = RangeConfig(),
) -> pd.DataFrame:
    # Build deterministic expected/floor/ceiling next-five production ranges.
    """Return short-term expected, floor, and ceiling ranges for PTS/AST/REB."""
    df = add_recent_rolling_volatility(performance_df, config=config)
    output_cols = ["player_id", "as_of_date", "game_id", "season", "team_id", "split"]
    for stat, stat_config in STAT_CONFIG.items():
        expected_col = f"expected_next_5_{stat}_avg"
        floor_col = f"floor_next_5_{stat}_avg"
        ceiling_col = f"ceiling_next_5_{stat}_avg"
        width_col = f"range_width_next_5_{stat}_avg"
        std_col = f"{stat}_rolling_std_{config.rolling_std_window}"
        target_col = STAT_TARGETS[stat]
        df[expected_col] = (
            config.season_avg_weight * df[stat_config["season_avg"]]
            + config.last_10_weight * df[stat_config["last_10"]]
            + config.last_5_weight * df[stat_config["last_5"]]
        )
        fallback_std = df.groupby("season")[stat_config["actual"]].transform("std")
        volatility = df[std_col].fillna(fallback_std).fillna(0)
        df[floor_col] = (df[expected_col] - config.volatility_multiplier * volatility).clip(lower=0)
        df[ceiling_col] = df[expected_col] + config.volatility_multiplier * volatility
        df[width_col] = df[ceiling_col] - df[floor_col]
        output_cols.extend([expected_col, floor_col, ceiling_col, width_col, target_col])
    return df[output_cols].dropna(subset=list(STAT_TARGETS.values())).reset_index(drop=True)


def evaluate_floor_ceiling_signals(signals: pd.DataFrame) -> pd.DataFrame:
    # Evaluate deterministic range outputs against next-five targets.
    """Return split/stat evaluation for short-term floor-ceiling signals."""
    return evaluate_short_term_ranges(signals, STAT_TARGETS)

