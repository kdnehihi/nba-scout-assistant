from dataclasses import dataclass


@dataclass(frozen=True)
class LSTMTaskConfig:
    stat_col: str
    stat_avg_col: str
    target_col: str
    sequence_length: int


LSTM_TASK_CONFIG = {
    "pts": LSTMTaskConfig(
        stat_col="pts",
        stat_avg_col="pts_season_avg",
        target_col="target_next_5_pts",
        sequence_length=10,
    ),

    "ast": LSTMTaskConfig(
        stat_col="ast",
        stat_avg_col="ast_season_avg",
        target_col="target_next_5_ast",
        sequence_length=10,
    ),

    "reb": LSTMTaskConfig(
        stat_col="reb",
        stat_avg_col="reb_season_avg",
        target_col="target_next_5_reb",
        sequence_length=15,
    ),
}
