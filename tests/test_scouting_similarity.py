from __future__ import annotations

import pandas as pd

from scouting.similarity import build_similarity_base, find_replacement_candidates


def test_find_replacement_candidates_respects_basic_filters():
    role = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "player_name": ["Target Guard", "Cheap Guard", "Expensive Guard"],
            "season": ["2024-25", "2024-25", "2024-25"],
            "team_id": ["AAA", "BBB", "CCC"],
            "position": ["G", "G", "G"],
            "age": [28, 24, 30],
            "minutes": [2000, 1800, 1900],
            "usage_pct": [0.25, 0.24, 0.26],
            "points_per_100": [30, 29, 31],
            "assists_per_100": [8, 7.5, 8.5],
            "rebounds_per_100": [5, 5.1, 4.9],
            "true_shooting_pct": [0.58, 0.57, 0.59],
            "three_point_attempt_rate": [0.40, 0.39, 0.41],
            "free_throw_rate": [0.25, 0.24, 0.26],
            "turnover_rate": [0.12, 0.13, 0.11],
            "steal_rate": [1.2, 1.1, 1.3],
            "block_rate": [0.3, 0.2, 0.4],
            "defensive_rebound_rate": [0.10, 0.11, 0.09],
            "foul_rate": [0.04, 0.04, 0.05],
            "scoring_creation": [25, 24, 26],
            "playmaking": [6, 5.8, 6.2],
            "shooting": [0.52, 0.51, 0.53],
            "rim_pressure": [0.25, 0.24, 0.26],
            "rebounding": [5, 5.1, 4.9],
            "perimeter_defense": [1, 1, 1],
            "interior_defense": [0.5, 0.4, 0.6],
            "two_way_impact": [5, 4, 6],
        }
    )
    salary = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "season_start_year": [2024, 2024, 2024],
            "salary_usd": [20_000_000, 8_000_000, 30_000_000],
            "salary_cap_share": [0.15, 0.06, 0.22],
            "team_id": ["AAA", "BBB", "CCC"],
        }
    )

    base = build_similarity_base(role, salary)
    candidates = find_replacement_candidates(base, "Target Guard", season="2024-25", cheaper_only=True)

    assert len(candidates) == 1
    assert candidates["player_name"].iloc[0] == "Cheap Guard"
    assert candidates["salary_cap_share_gap"].iloc[0] < 0

