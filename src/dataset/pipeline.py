from __future__ import annotations

from .features_compensation import build_player_salary_history
from .features_long_term import build_long_term_inference, build_long_term_training
from .features_performance import build_performance_training
from .features_role import build_role_features
from .loaders import (
    DataPaths,
    load_player_game_logs,
    load_player_season_salaries,
    load_player_season_stats,
    load_players,
    load_salary_cap,
)


def build_all_gold_datasets(paths: DataPaths) -> dict[str, object]:
    # Orchestrate all dataset builders and persist gold outputs.
    """Build all gold datasets and write them to the gold data layer."""
    players = load_players(paths)
    game_logs = load_player_game_logs(paths)
    season_stats = load_player_season_stats(paths)
    salaries = load_player_season_salaries(paths)
    salary_cap = load_salary_cap(paths)

    role_features = build_role_features(players, season_stats)
    performance_training = build_performance_training(game_logs)
    salary_history = build_player_salary_history(salaries, salary_cap, players)
    long_term_training = build_long_term_training(game_logs, players, season_stats)
    long_term_inference = build_long_term_inference(game_logs, players, season_stats)

    outputs = {
        "player_role_features_clean": role_features,
        "performance_training_clean": performance_training,
        "player_salary_history_clean": salary_history,
        "long_term_player_forecast_training": long_term_training,
        "long_term_player_forecast_inference": long_term_inference,
    }
    paths.gold_dir.mkdir(parents=True, exist_ok=True)
    for name, dataframe in outputs.items():
        dataframe.to_parquet(paths.gold_dir / f"{name}.parquet", index=False)
    return outputs
