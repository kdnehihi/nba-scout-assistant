from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.evaluation.evaluate_similarity import (
    build_future_similarity_ground_truth,
    build_profile_clusters,
)
from src.pipelines.recommendation import (
    RecommendationPipelineData,
    build_recommended_player_detail,
    evaluate_recommendation_result,
    recommend_similar_players,
)
from src.scouting.ranking import RecommendationRankerArtifact, SeasonFeaturePreprocessor
from src.scouting.recommendation import (
    ALL_RECOMMENDER_FEATURES,
    build_recommendation_base,
)
from tests.test_scouting_recommendation import sample_role_features


def make_recommendation_pipeline_data() -> RecommendationPipelineData:
    current = sample_role_features()
    future = current.copy()
    future["season"] = "2025-26"
    future.loc[future["player_name"].eq("Target Guard"), ["points_per_100", "assists_per_100"]] = [34, 9]
    future.loc[future["player_name"].eq("Similar Guard"), ["points_per_100", "assists_per_100"]] = [33.8, 8.8]
    base = build_recommendation_base(pd.concat([current, future], ignore_index=True))
    salary_history = pd.DataFrame(
        {
            "player_id": [2],
            "player_name": ["Similar Guard"],
            "season_start_year": [2024],
            "season_label": ["2024-25"],
            "salary_usd": [12_000_000],
        }
    )
    return RecommendationPipelineData(
        recommendation_base=base,
        salary_history=salary_history,
        contract_history=pd.DataFrame(),
        profile_clusters=build_profile_clusters(base, ALL_RECOMMENDER_FEATURES, target_cluster_size=2),
        future_ground_truth=build_future_similarity_ground_truth(base, ALL_RECOMMENDER_FEATURES, relevant_n=2),
    )


def test_recommendation_pipeline_ranks_and_evaluates_candidates():
    pipeline_data = make_recommendation_pipeline_data()

    recommendations = recommend_similar_players(
        pipeline_data,
        player_name="Target Guard",
        season="2024-25",
        top_n=2,
    )
    diagnostics = evaluate_recommendation_result(pipeline_data, recommendations, top_n=2)

    assert recommendations["player_name"].iloc[0] == "Similar Guard"
    assert "cluster_same_cluster_rate" in diagnostics
    assert "future_proxy_recall_at_k" in diagnostics


def test_recommended_player_detail_uses_salary_context():
    pipeline_data = make_recommendation_pipeline_data()

    context = build_recommended_player_detail(pipeline_data, player_id=2, player_name="Similar Guard")

    assert context["latest_salary"]["salary_usd"] == 12_000_000


def test_recommendation_pipeline_supports_selected_ranker_and_fallback():
    fallback_data = make_recommendation_pipeline_data()
    fallback = recommend_similar_players(fallback_data, "Target Guard", season="2024-25", top_n=2)
    preprocessor = SeasonFeaturePreprocessor().fit(fallback_data.recommendation_base)
    artifact_data = replace(
        fallback_data,
        recommendation_base=preprocessor.transform(fallback_data.recommendation_base),
        ranker_artifact=RecommendationRankerArtifact(
            algorithm="season_normalized_euclidean",
            preprocessor=preprocessor,
            version="test-ranker",
        ),
    )
    ranked = recommend_similar_players(artifact_data, "Target Guard", season="2024-25", top_n=2)

    assert {"ranking_algorithm", "ranker_version"}.issubset(fallback.columns)
    assert fallback["ranking_algorithm"].eq("season_normalized_euclidean").all()
    assert ranked["ranker_version"].eq("test-ranker").all()
