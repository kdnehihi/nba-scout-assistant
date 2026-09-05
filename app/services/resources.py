from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipelines.artifacts import (
    load_long_term_model_artifacts,
    load_recommendation_ranker_artifact,
    load_short_term_model_artifacts,
)
from src.pipelines.forecasting import (
    load_long_term_prediction_data,
    load_short_term_prediction_data,
)
from src.pipelines.recommendation import (
    RecommendationPipelineData,
    load_recommendation_pipeline_data,
)


@dataclass(frozen=True)
class AppResources:
    recommendation_data: RecommendationPipelineData
    short_term_data: pd.DataFrame
    long_term_data: pd.DataFrame
    short_term_models: dict[str, Any]
    long_term_models: dict[tuple[str, int], Any]


def load_app_resources(
    data_dir: str | Path = "data",
    artifact_dir: str | Path = "artifacts",
) -> AppResources:
    """Load data and model artifacts used by API services."""
    recommendation_ranker = load_recommendation_ranker_artifact(artifact_dir, required=False)
    return AppResources(
        recommendation_data=load_recommendation_pipeline_data(
            data_dir,
            ranker_artifact=recommendation_ranker,
        ),
        short_term_data=load_short_term_prediction_data(data_dir),
        long_term_data=load_long_term_prediction_data(data_dir),
        short_term_models=load_short_term_model_artifacts(artifact_dir),
        long_term_models=load_long_term_model_artifacts(artifact_dir),
    )
