from __future__ import annotations

import re

import pandas as pd

try:
    from src.config.long_term_config import LongTermTaskConfig
except ModuleNotFoundError:
    from config.long_term_config import LongTermTaskConfig

from .splits import assign_long_term_temporal_split


LONG_TERM_BASE_COLUMNS = (
    "player_id",
    "anchor_season",
    "anchor_season_start_year",
    "split",
)

LONG_TERM_ALLOWED_SPLITS = ("train", "validation", "test")
LONG_TERM_TARGET_PATTERN = re.compile(r"_(h[123])$")
LONG_TERM_METADATA_COLUMNS = {
    "player_id",
    "player_name",
    "anchor_season",
    "anchor_season_start_year",
    "anchor_date",
    "split",
}
# Future-horizon labels are excluded from model features to prevent leakage.
# Some labels, such as per-100-possession targets, are retained for research
# artifacts even when they are not selected production forecast tasks.
LONG_TERM_EXCLUDED_PREFIXES = (
    "active_h",
    "games_played_h",
    "minutes_per_game_h",
    "pts_per_36_h",
    "ast_per_36_h",
    "reb_per_36_h",
    "pts_per_100_h",
    "ast_per_100_h",
    "reb_per_100_h",
    "pts_per_game_h",
    "ast_per_game_h",
    "reb_per_game_h",
    "age_at_h",
)


def is_long_term_model_feature(column: str) -> bool:
    """Return True when a column is allowed as an anchor-time model feature."""
    if column in LONG_TERM_METADATA_COLUMNS:
        return False
    if column.startswith(LONG_TERM_EXCLUDED_PREFIXES):
        return False
    if LONG_TERM_TARGET_PATTERN.search(column):
        return False
    return True


def infer_long_term_feature_columns(df: pd.DataFrame) -> list[str]:
    """Infer non-leaking long-term features from a gold long-term dataframe."""
    return [column for column in df.columns if is_long_term_model_feature(column)]


def validate_long_term_columns(
    df: pd.DataFrame,
    task_config: LongTermTaskConfig,
    feature_cols: list[str],
) -> None:
    """Validate that long-term data has all columns required for one task."""
    required_columns = set(LONG_TERM_BASE_COLUMNS) | set(feature_cols) | {task_config.target_col}
    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise KeyError(f"Missing long-term columns for {task_config.task} h{task_config.horizon}: {missing}")


def validate_long_term_split(
    df: pd.DataFrame,
    split_col: str = "split",
    season_col: str = "anchor_season",
) -> None:
    """Validate split labels and check they match the configured long-term split map."""
    if split_col not in df.columns:
        raise KeyError(f"Missing split column: {split_col}")
    if season_col not in df.columns:
        raise KeyError(f"Missing season column: {season_col}")

    observed = set(df[split_col].dropna().astype(str))
    unexpected = sorted(observed - set(LONG_TERM_ALLOWED_SPLITS))
    if unexpected:
        raise ValueError(f"Unexpected long-term split labels: {unexpected}")

    expected_split = df[season_col].map(assign_long_term_temporal_split)
    mismatch = df[split_col].astype(str).ne(expected_split.astype(str))
    if mismatch.any():
        sample = df.loc[mismatch, [season_col, split_col]].head(5).to_dict("records")
        raise ValueError(f"Long-term split does not match configured season map. Sample mismatches: {sample}")

    missing_splits = sorted(set(LONG_TERM_ALLOWED_SPLITS) - observed)
    if missing_splits:
        raise ValueError(f"Missing long-term split labels: {missing_splits}")


def prepare_long_term_training(
    df: pd.DataFrame,
    task_config: LongTermTaskConfig,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Return clean long-term rows for one task after schema and target checks."""
    validate_long_term_columns(df, task_config=task_config, feature_cols=feature_cols)
    validate_long_term_split(df)

    required_columns = [*LONG_TERM_BASE_COLUMNS, task_config.target_col]
    before = len(df)

    long_term_df = df.dropna(subset=required_columns).copy()
    long_term_df = long_term_df.sort_values(["anchor_season_start_year", "player_id"]).reset_index(drop=True)

    print(f"Dropped rows for {task_config.task} h{task_config.horizon}: {before:,} -> {len(long_term_df):,}")

    return long_term_df


def prepare_long_term_modeling_data(
    df: pd.DataFrame,
    task_config: LongTermTaskConfig,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare long-term model data and return the dataframe plus selected features."""
    selected_feature_cols = feature_cols or infer_long_term_feature_columns(df)
    prepared_df = prepare_long_term_training(
        df,
        task_config=task_config,
        feature_cols=selected_feature_cols,
    )
    return prepared_df, selected_feature_cols
