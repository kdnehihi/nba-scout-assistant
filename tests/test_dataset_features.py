from __future__ import annotations

import pandas as pd

from dataset.features_long_term import build_long_term_training
from dataset.features_performance import build_performance_training
from dataset.features_role import build_role_features
from dataset.features_salary import build_salary_training


def sample_players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["Player One", "Player Two"],
            "birth_date": ["1995-01-01", "1997-01-01"],
            "position": ["G", "F"],
            "height": [76, 80],
            "weight": [200, 220],
        }
    )


def sample_season_stats() -> pd.DataFrame:
    rows = []
    for season in ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]:
        rows.append(
            {
                "player_id": 1,
                "player_name": "Player One",
                "season": season,
                "team_id": "AAA",
                "age": 25,
                "position": "G",
                "minutes": 1800,
                "usage_pct": 0.24,
                "points_per_100": 32,
                "assists_per_100": 8,
                "rebounds_per_100": 7,
                "true_shooting_pct": 0.58,
                "three_point_attempt_rate": 0.42,
                "free_throw_rate": 0.28,
                "turnover_rate": 0.12,
                "steal_rate": 1.5,
                "block_rate": 0.4,
                "defensive_rebound_rate": 0.12,
                "foul_rate": 0.04,
                "pace": 100,
                "possessions": 3500,
                "offensive_rating": 115,
                "defensive_rating": 110,
            }
        )
    return pd.DataFrame(rows)


def sample_game_logs() -> pd.DataFrame:
    rows = []
    seasons = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
    game_id = 1
    for season_idx, season in enumerate(seasons):
        for game_number in range(12):
            rows.append(
                {
                    "player_id": 1,
                    "player_name": "Player One",
                    "game_id": game_id,
                    "game_date": f"{2016 + season_idx}-11-{game_number + 1:02d}",
                    "season": season,
                    "team_id": "AAA",
                    "opponent": "BBB",
                    "home_away": "HOME",
                    "rest_days": 2,
                    "minutes": 30 + game_number % 3,
                    "points": 10 + game_number,
                    "assists": 3 + game_number % 4,
                    "rebounds": 5 + game_number % 5,
                    "fga": 12,
                    "fta": 4,
                    "fg3a": 5,
                    "true_shooting_pct": 0.56,
                }
            )
            game_id += 1
    return pd.DataFrame(rows)


def test_build_role_features_creates_role_dimensions():
    role = build_role_features(sample_players(), sample_season_stats())

    assert "scoring_creation" in role.columns
    assert "two_way_impact" in role.columns
    assert role["position"].iloc[0] == "G"


def test_build_performance_training_creates_next_five_targets():
    performance = build_performance_training(sample_game_logs())

    assert {"target_next_5_pts_avg", "pts_last_5", "split"}.issubset(performance.columns)
    assert set(performance["split"]).issubset({"train", "validation", "test"})
    assert performance["target_next_5_pts_avg"].notna().all()


def test_build_salary_training_merges_cap_players_and_role_features():
    role = build_role_features(sample_players(), sample_season_stats())
    salaries = pd.DataFrame(
        {
            "player_name": ["Player One"],
            "team": ["AAA"],
            "season_start_year": [2022],
            "season_end_year": [2023],
            "season_label": ["2022-23"],
            "salary_usd": [10_000_000],
            "source": ["test"],
            "source_file": ["test.csv"],
            "collected_at": ["2026-01-01"],
        }
    )
    salary_cap = pd.DataFrame({"season": ["2022-23"], "salary_cap_usd": [100_000_000], "tax_level_usd": [120_000_000]})

    salary = build_salary_training(salaries, salary_cap, sample_players(), role)

    assert salary["salary_cap_share"].iloc[0] == 0.1
    assert "scoring_creation" in salary.columns
    assert salary["split"].iloc[0] == "train"


def test_build_long_term_training_creates_horizon_targets():
    long_term = build_long_term_training(sample_game_logs(), sample_players(), sample_season_stats())

    assert {"active_h1", "pts_per_36_h1", "low_availability_h1", "split"}.issubset(long_term.columns)
    assert set(long_term["split"]).issubset({"train", "validation", "test"})
