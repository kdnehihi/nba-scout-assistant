from __future__ import annotations

import pandas as pd

from dataset.loaders import (
    DataPaths,
    load_performance_training_clean,
    load_role_features_clean,
    load_salary_training_clean,
)

from .ranges import build_short_term_floor_ceiling_signals, evaluate_floor_ceiling_signals
from .signals import build_player_consistency_signals, build_player_trend_signals
from .similarity import build_similarity_base, find_replacement_candidates


def build_all_scouting_artifacts(paths: DataPaths, example_queries: list[dict[str, object]] | None = None) -> dict[str, pd.DataFrame]:
    # Build deterministic scouting artifacts from clean gold datasets.
    """Build and persist deterministic scouting artifacts in the gold layer."""
    performance = load_performance_training_clean(paths)
    role_features = load_role_features_clean(paths)
    salary = load_salary_training_clean(paths)

    trend_signals = build_player_trend_signals(performance)
    consistency_signals = build_player_consistency_signals(performance)
    floor_ceiling_signals = build_short_term_floor_ceiling_signals(performance)
    floor_ceiling_evaluation = evaluate_floor_ceiling_signals(floor_ceiling_signals)

    outputs = {
        "player_trend_signals": trend_signals,
        "player_consistency_signals": consistency_signals,
        "short_term_floor_ceiling_signals": floor_ceiling_signals,
        "short_term_floor_ceiling_evaluation": floor_ceiling_evaluation,
    }

    if example_queries:
        similarity_base = build_similarity_base(role_features, salary)
        examples = []
        for query in example_queries:
            examples.append(find_replacement_candidates(similarity_base, **query))
        if examples:
            outputs["replacement_candidate_examples"] = pd.concat(examples, ignore_index=True)

    paths.gold_dir.mkdir(parents=True, exist_ok=True)
    for name, dataframe in outputs.items():
        dataframe.to_parquet(paths.gold_dir / f"{name}.parquet", index=False)
    return outputs

