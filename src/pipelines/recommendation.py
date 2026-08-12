from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.dataset.loaders import (
    DataPaths,
    load_contract_history,
    load_performance_training_clean,
    load_player_salary_history_clean,
    load_role_features_clean,
    resolve_data_paths,
)
from src.evaluation.evaluate_similarity import (
    build_future_similarity_ground_truth,
    build_profile_clusters,
    evaluate_recommendations_against_ground_truth,
    recommendation_cluster_agreement,
)
from src.scouting.compensation import build_player_compensation_context
from src.scouting.recommendation import ALL_RECOMMENDER_FEATURES, build_recommendation_base, recommend_players


@dataclass(frozen=True)
class RecommendationPipelineData:
    """Materialized dataframes used by the recommendation pipeline."""

    recommendation_base: pd.DataFrame
    salary_history: pd.DataFrame
    contract_history: pd.DataFrame
    profile_clusters: pd.DataFrame
    future_ground_truth: pd.DataFrame


def load_recommendation_pipeline_data(
    data_dir: Path | str = "data",
    cluster_target_size: int = 8,
) -> RecommendationPipelineData:
    # Load gold datasets and construct recommendation evaluation artifacts.
    """Load all data required for player recommendation and detail context."""
    paths: DataPaths = resolve_data_paths(data_dir)
    role_features = load_role_features_clean(paths)
    performance = load_performance_training_clean(paths)
    salary_history = load_player_salary_history_clean(paths)
    contract_history = load_contract_history(paths, required=False)

    recommendation_base = build_recommendation_base(
        role_df=role_features,
        performance_df=performance,
    )
    profile_clusters = build_profile_clusters(
        recommendation_base,
        features=ALL_RECOMMENDER_FEATURES,
        target_cluster_size=cluster_target_size,
    )
    future_ground_truth = build_future_similarity_ground_truth(
        recommendation_base,
        features=ALL_RECOMMENDER_FEATURES,
    )
    return RecommendationPipelineData(
        recommendation_base=recommendation_base,
        salary_history=salary_history,
        contract_history=contract_history,
        profile_clusters=profile_clusters,
        future_ground_truth=future_ground_truth,
    )


def recommend_similar_players(
    pipeline_data: RecommendationPipelineData,
    player_name: str,
    season: str | None = None,
    top_n: int = 5,
    preset: str = "playing_profile",
    same_season: bool = True,
    same_position_group: bool = True,
    minutes_min: float | None = 500,
) -> pd.DataFrame:
    # Rank similar players for one query using the deterministic recommender.
    """Return top-K similar players for one target player query."""
    return recommend_players(
        base_df=pipeline_data.recommendation_base,
        player_name=player_name,
        season=season,
        top_n=top_n,
        preset=preset,
        same_season=same_season,
        same_position_group=same_position_group,
        minutes_min=minutes_min,
    )


def evaluate_recommendation_result(
    pipeline_data: RecommendationPipelineData,
    recommendations: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, float]:
    # Evaluate one recommendation output with cluster and future-outcome proxies.
    """Return diagnostic metrics for a recommendation result."""
    cluster_metrics = recommendation_cluster_agreement(
        recommendations,
        pipeline_data.profile_clusters,
        top_n=top_n,
    )
    ground_truth_metrics = evaluate_recommendations_against_ground_truth(
        recommendations,
        pipeline_data.future_ground_truth,
        top_n=top_n,
    )
    return {
        **{f"cluster_{key}": value for key, value in cluster_metrics.items()},
        **{f"future_proxy_{key}": value for key, value in ground_truth_metrics.items()},
    }


def build_recommended_player_detail(
    pipeline_data: RecommendationPipelineData,
    player_id: int | str | None = None,
    player_name: str | None = None,
) -> dict[str, Any]:
    # Build the detail-page compensation context for a recommended player.
    """Return salary and contract context for one player."""
    return build_player_compensation_context(
        salary_history=pipeline_data.salary_history,
        player_id=player_id,
        player_name=player_name,
        contract_history=pipeline_data.contract_history,
    )


def build_recommendation_report(
    data_dir: Path | str,
    player_name: str,
    season: str | None = None,
    top_n: int = 5,
    preset: str = "playing_profile",
) -> dict[str, Any]:
    # Build one product-ready recommendation report for API or notebook display.
    """Return recommendations, diagnostics, and candidate compensation context."""
    pipeline_data = load_recommendation_pipeline_data(data_dir)
    recommendations = recommend_similar_players(
        pipeline_data=pipeline_data,
        player_name=player_name,
        season=season,
        top_n=top_n,
        preset=preset,
    )
    diagnostics = evaluate_recommendation_result(
        pipeline_data=pipeline_data,
        recommendations=recommendations,
        top_n=top_n,
    )
    candidate_context = {
        str(row["player_id"]): build_recommended_player_detail(
            pipeline_data,
            player_id=row["player_id"],
            player_name=row["player_name"],
        )
        for _, row in recommendations.iterrows()
    }
    return {
        "recommendations": recommendations,
        "diagnostics": diagnostics,
        "candidate_context": candidate_context,
    }
