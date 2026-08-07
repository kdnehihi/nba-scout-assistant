from __future__ import annotations
import pandas as pd
from src.dataset.loaders import (
    DataPaths,
    load_player_game_logs
)

from src.dataset.features_performance import (
    build_performance_training
)


LSTM_TASK_CONFIG = {
    "pts": {
        "stat_col": "pts",
        "stat_avg_col": "pts_season_avg",
        "target_col": "target_next_5_pts"
    },
    "ast": {
        "stat_col": "ast",
        "stat_avg_col": "ast_season_avg",
        "target_col": "target_next_5_ast"
    },
    "reb": {
        "stat_col": "reb",
        "stat_avg_col": "reb_season_avg",
        "target_col": "target_next_5_reb"
    }
}

def prepare_sequence_training(df:pd.DataFrame, task_config: dict) -> pd.DataFrame:
    sequence_df = df.copy()
    required_columns = [
        "player_id",
        "season",
        "as_of_date",
        "game_id",
        "split",
        "min",
        "min_season_avg",

        task_config["stat_col"],
        task_config["stat_avg_col"],
        task_config["target_col"],
    ]
    missing = set(required_columns) - set(sequence_df.columns())
    if missing:
        raise KeyError(f"Missing columns: {missing}") 
    before_len = len(sequence_df)
    sequence_df = (sequence_df.drop_na(subset=required_columns)).copy()
    sequence_df = sequence_df.sort_values(["player_id", "season", "as_of_date", "game_id"]).reset_index(drop=True)
    print(
        f"Dropped rows: {before:,} -> {len(sequence_df):,}"
    )


    return sequence_df

