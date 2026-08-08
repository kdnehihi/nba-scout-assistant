from dataclasses import dataclass


@dataclass(frozen=True)
class LSTMTaskConfig:
    stat_col: str
    stat_avg_col: str
    target_col: str
    sequence_length: int
    hidden_size: int = 64
    batch_size: int = 128
    learning_rate: float = 5e-4
    dropout: float = 0.15
    scale_target_delta: bool = False


LSTM_TASK_CONFIG = {
    "points": LSTMTaskConfig(
        stat_col="pts",
        stat_avg_col="pts_season_avg",
        target_col="target_next_5_pts_avg",
        sequence_length=10,
    ),

    "assists": LSTMTaskConfig(
        stat_col="ast",
        stat_avg_col="ast_season_avg",
        target_col="target_next_5_ast_avg",
        sequence_length=10,
    ),

    "rebounds": LSTMTaskConfig(
        stat_col="reb",
        stat_avg_col="reb_season_avg",
        target_col="target_next_5_reb_avg",
        sequence_length=15,
    ),
}

LSTM_TASK_ALIASES = {
    "pts": "points",
    "ast": "assists",
    "reb": "rebounds",
}


def resolve_lstm_task_config(task: str) -> tuple[str, LSTMTaskConfig]:
    """Return canonical task name and config for a short-term LSTM task."""
    canonical_task = LSTM_TASK_ALIASES.get(task, task)
    if canonical_task not in LSTM_TASK_CONFIG:
        raise KeyError(f"Unknown LSTM task: {task}")
    return canonical_task, LSTM_TASK_CONFIG[canonical_task]
