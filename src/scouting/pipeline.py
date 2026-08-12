from __future__ import annotations

import pandas as pd

from src.dataset.loaders import (
    DataPaths,
    load_performance_training_clean,
    load_player_salary_history_clean,
    load_role_features_clean,
)
from src.evaluation.evaluate_similarity import (
    build_future_similarity_ground_truth,
    build_profile_clusters,
    evaluate_recommendations_against_ground_truth,
    recommendation_cluster_agreement,
)

from .ranges import build_short_term_floor_ceiling_signals, evaluate_floor_ceiling_signals
from .recommendation import ALL_RECOMMENDER_FEATURES, build_recommendation_base, recommend_players
from .signals import build_player_consistency_signals, build_player_trend_signals


def build_all_scouting_artifacts(paths: DataPaths, example_queries: list[dict[str, object]] | None = None) -> dict[str, pd.DataFrame]:
    # Build deterministic scouting artifacts from clean gold datasets.
    """Build and persist deterministic scouting artifacts in the gold layer."""
    performance = load_performance_training_clean(paths)
    role_features = load_role_features_clean(paths)
    salary_history = load_player_salary_history_clean(paths)

    trend_signals = build_player_trend_signals(performance)
    consistency_signals = build_player_consistency_signals(performance)
    floor_ceiling_signals = build_short_term_floor_ceiling_signals(performance)
    floor_ceiling_evaluation = evaluate_floor_ceiling_signals(floor_ceiling_signals)
    recommendation_base = build_recommendation_base(role_features, performance_df=performance)
    recommendation_clusters = build_profile_clusters(
        recommendation_base,
        features=ALL_RECOMMENDER_FEATURES,
    )
    recommendation_ground_truth = build_future_similarity_ground_truth(
        recommendation_base,
        features=ALL_RECOMMENDER_FEATURES,
    )

    outputs = {
        "player_trend_signals": trend_signals,
        "player_consistency_signals": consistency_signals,
        "short_term_floor_ceiling_signals": floor_ceiling_signals,
        "short_term_floor_ceiling_evaluation": floor_ceiling_evaluation,
        "player_recommendation_base": recommendation_base,
        "player_recommendation_profile_clusters": recommendation_clusters,
        "player_recommendation_ground_truth": recommendation_ground_truth,
        "player_salary_history_context": salary_history,
    }

    if example_queries:
        examples = []
        diagnostics = []
        for query in example_queries:
            recommendations = recommend_players(recommendation_base, **query)
            examples.append(recommendations)
            diagnostics.append(
                {
                    **recommendation_cluster_agreement(recommendations, recommendation_clusters),
                    **{
                        f"ground_truth_{key}": value
                        for key, value in evaluate_recommendations_against_ground_truth(
                            recommendations,
                            recommendation_ground_truth,
                        ).items()
                    },
                }
            )
        if examples:
            outputs["player_recommendation_examples"] = pd.concat(examples, ignore_index=True)
        if diagnostics:
            outputs["player_recommendation_cluster_diagnostics"] = pd.DataFrame(diagnostics)

    paths.gold_dir.mkdir(parents=True, exist_ok=True)
    for name, dataframe in outputs.items():
        dataframe.to_parquet(paths.gold_dir / f"{name}.parquet", index=False)
    return outputs
