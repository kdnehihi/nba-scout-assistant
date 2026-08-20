from __future__ import annotations

import pandas as pd
import pytest

from evaluation.evaluate_similarity import (
    build_future_similarity_ground_truth,
    build_profile_clusters,
    evaluate_recommendations_against_ground_truth,
    recommendation_cluster_agreement,
)
from scouting.recommendation import (
    ALL_RECOMMENDER_FEATURES,
    build_recommendation_base,
    position_group,
    recommend_players,
    select_target_row,
)


def sample_role_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "player_name": ["Target Guard", "Similar Guard", "Different Guard", "Big Player"],
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "team_id": ["AAA", "BBB", "CCC", "DDD"],
            "position": ["G", "G", "G", "C"],
            "height": [76, 77, 74, 83],
            "weight": [205, 207, 190, 250],
            "minutes": [2200, 2100, 900, 2300],
            "usage_pct": [0.25, 0.245, 0.16, 0.22],
            "points_per_100": [31, 30.5, 15, 24],
            "assists_per_100": [8, 7.8, 3, 2],
            "rebounds_per_100": [5, 5.2, 4, 14],
            "true_shooting_pct": [0.59, 0.585, 0.52, 0.61],
            "three_point_attempt_rate": [0.42, 0.41, 0.20, 0.05],
            "free_throw_rate": [0.25, 0.24, 0.12, 0.35],
            "turnover_rate": [0.12, 0.13, 0.18, 0.15],
            "steal_rate": [1.2, 1.1, 0.8, 0.7],
            "block_rate": [0.3, 0.3, 0.2, 2.4],
            "defensive_rebound_rate": [0.10, 0.11, 0.08, 0.28],
            "foul_rate": [0.04, 0.04, 0.06, 0.05],
            "scoring_creation": [25, 24.5, 12, 20],
            "playmaking": [6.0, 5.9, 2.5, 1.8],
            "shooting": [0.53, 0.52, 0.35, 0.30],
            "rim_pressure": [0.25, 0.245, 0.12, 0.30],
            "rebounding": [5.0, 5.1, 4.0, 14.0],
            "perimeter_defense": [1.0, 1.0, 0.7, 0.5],
            "interior_defense": [0.5, 0.5, 0.3, 2.5],
            "two_way_impact": [5.0, 4.8, -2.0, 3.0],
        }
    )


def test_recommend_players_returns_grouped_explanations():
    base = build_recommendation_base(sample_role_features())

    recommendations = recommend_players(
        base,
        player_name="Target Guard",
        season="2024-25",
        top_n=2,
        minutes_min=500,
    )

    assert recommendations["player_name"].iloc[0] == "Similar Guard"
    assert "matched groups" in recommendations["recommendation_reason"].iloc[0]
    assert recommendations["position_group"].eq("guard").all()


def test_scoring_profile_ranks_by_observed_scoring_similarity():
    base = build_recommendation_base(sample_role_features())

    recommendations = recommend_players(
        base,
        player_name="Target Guard",
        season="2024-25",
        top_n=2,
        preset="scoring_profile",
        minutes_min=500,
    )

    assert recommendations["player_name"].iloc[0] == "Similar Guard"
    assert recommendations["recommendation_score"].equals(recommendations["scoring_profile_score"])


def test_defensive_profile_ranks_by_observed_defensive_similarity():
    base = build_recommendation_base(sample_role_features())

    recommendations = recommend_players(
        base,
        player_name="Target Guard",
        season="2024-25",
        top_n=2,
        preset="defensive_profile",
        minutes_min=500,
    )

    assert recommendations["player_name"].iloc[0] == "Similar Guard"
    assert recommendations["recommendation_score"].equals(recommendations["defensive_profile_score"])


def test_recommendation_base_collapses_multi_team_seasons():
    role_features = sample_role_features()
    second_stint = role_features.loc[[1]].copy()
    second_stint["team_id"] = "EEE"
    second_stint["minutes"] = 400
    second_stint["points_per_100"] = 27.0

    base = build_recommendation_base(pd.concat([role_features, second_stint], ignore_index=True))
    similar_guard = base[base["player_id"].eq(2)].iloc[0]

    assert base["player_id"].eq(2).sum() == 1
    assert similar_guard["team_id"] == "MULTI"
    assert similar_guard["minutes"] == 2500
    assert similar_guard["points_per_100"] == pytest.approx((30.5 * 2100 + 27.0 * 400) / 2500)


def test_select_target_row_uses_normalized_player_name():
    role_features = sample_role_features()
    role_features.loc[0, "player_name"] = "Luka Dončić"
    base = build_recommendation_base(role_features)

    target = select_target_row(base, player_name="luka doncic", season="2024-25")

    assert target["player_id"] == 1


def test_profile_cluster_agreement_diagnostic_runs_on_recommendations():
    base = build_recommendation_base(sample_role_features())
    recommendations = recommend_players(base, "Target Guard", season="2024-25", top_n=2, minutes_min=500)
    clusters = build_profile_clusters(
        base,
        features=ALL_RECOMMENDER_FEATURES,
        target_cluster_size=2,
        random_state=42,
    )

    diagnostics = recommendation_cluster_agreement(recommendations, clusters, top_n=2)

    assert diagnostics["rows"] == 2.0
    assert 0.0 <= diagnostics["same_cluster_rate"] <= 1.0
    assert diagnostics["target_cluster_size"] >= 1.0


def test_future_similarity_ground_truth_scores_recommendation_hits():
    current = sample_role_features()
    future = current.copy()
    future["season"] = "2025-26"
    future.loc[future["player_name"].eq("Target Guard"), ["points_per_100", "assists_per_100"]] = [34, 9]
    future.loc[future["player_name"].eq("Similar Guard"), ["points_per_100", "assists_per_100"]] = [33.8, 8.8]
    future.loc[future["player_name"].eq("Different Guard"), ["points_per_100", "assists_per_100"]] = [18, 3]
    base = build_recommendation_base(pd.concat([current, future], ignore_index=True))

    recommendations = recommend_players(base, "Target Guard", season="2024-25", top_n=2, minutes_min=500)
    ground_truth = build_future_similarity_ground_truth(
        base,
        features=ALL_RECOMMENDER_FEATURES,
        relevant_n=2,
        minutes_min=500,
    )
    metrics = evaluate_recommendations_against_ground_truth(recommendations, ground_truth, top_n=2)

    assert "Similar Guard" in ground_truth["player_name"].tolist()
    assert metrics["hit_count"] >= 1.0
    assert 0.0 <= metrics["recall_at_k"] <= 1.0


def test_position_group_maps_common_position_shapes():
    assert position_group("PG/SG") == "guard"
    assert position_group("SF/PF") == "wing"
    assert position_group("C") == "big"
