from __future__ import annotations

import pandas as pd

from scouting.ranges import build_short_term_floor_ceiling_signals, evaluate_floor_ceiling_signals
from scouting.signals import build_player_consistency_signals, build_player_trend_signals


def sample_performance_training() -> pd.DataFrame:
    rows = []
    for game_number in range(12):
        pts = 10 + game_number
        ast = 3 + game_number % 3
        reb = 5 + game_number % 4
        minutes = 25 + game_number % 5
        rows.append(
            {
                "player_id": 1,
                "as_of_date": f"2024-01-{game_number + 1:02d}",
                "game_id": game_number + 1,
                "season": "2023-24",
                "team_id": "AAA",
                "split": "validation",
                "pts": pts,
                "ast": ast,
                "reb": reb,
                "min": minutes,
                "pts_last_5": pts - 2,
                "pts_last_10": pts - 3,
                "pts_season_avg": 12,
                "ast_last_5": ast,
                "ast_last_10": ast,
                "ast_season_avg": 3,
                "reb_last_5": reb,
                "reb_last_10": reb,
                "reb_season_avg": 5,
                "min_last_5": minutes,
                "min_last_10": minutes,
                "min_season_avg": 25,
                "target_next_5_pts_avg": pts + 1,
                "target_next_5_ast_avg": ast + 0.5,
                "target_next_5_reb_avg": reb + 0.5,
            }
        )
    return pd.DataFrame(rows)


def test_build_player_trend_signals_returns_latest_player_season_row():
    trend = build_player_trend_signals(sample_performance_training())

    assert len(trend) == 1
    assert trend["overall_trend"].iloc[0] in {"improving", "stable", "declining"}
    assert trend["as_of_date"].iloc[0] == "2024-01-12"


def test_build_player_consistency_signals_returns_volatility_label():
    consistency = build_player_consistency_signals(sample_performance_training(), min_games=10)

    assert len(consistency) == 1
    assert consistency["consistency_label"].iloc[0] in {"consistent", "balanced", "volatile"}
    assert consistency["games_observed"].iloc[0] == 12


def test_build_short_term_floor_ceiling_signals_and_evaluation():
    signals = build_short_term_floor_ceiling_signals(sample_performance_training())
    evaluation = evaluate_floor_ceiling_signals(signals)

    assert {"expected_next_5_pts_avg", "floor_next_5_pts_avg", "ceiling_next_5_pts_avg"}.issubset(signals.columns)
    assert (signals["floor_next_5_pts_avg"] <= signals["ceiling_next_5_pts_avg"]).all()
    assert set(evaluation["stat"]) == {"pts", "ast", "reb"}
    assert evaluation["coverage_rate"].between(0, 1).all()

