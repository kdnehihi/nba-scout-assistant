from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.config.recommendation_config import (
    DEFAULT_SHRINKAGE_PRIOR_STRENGTH,
    PAIR_FEATURES,
    RECOMMENDATION_FEATURES,
    RECOMMENDATION_RANKER_VERSION,
    SHRINKAGE_FEATURES,
)


def normalized_feature_name(feature: str) -> str:
    """Return the stable column name for a season-normalized feature."""
    return f"normalized__{feature}"


NORMALIZED_RECOMMENDATION_FEATURES = tuple(
    normalized_feature_name(feature) for feature in RECOMMENDATION_FEATURES
)


@dataclass
class SeasonFeaturePreprocessor:
    """Shrink noisy rates and normalize profiles against each complete season pool."""

    prior_strength: float = DEFAULT_SHRINKAGE_PRIOR_STRENGTH
    features: tuple[str, ...] = RECOMMENDATION_FEATURES
    shrinkage_features: tuple[str, ...] = SHRINKAGE_FEATURES
    season_statistics: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    def _validate(self, df: pd.DataFrame) -> None:
        required = {"season", "minutes", *self.features}
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"Missing recommendation preprocessing columns: {missing}")

    def _calculate_season_statistics(self, season_df: pd.DataFrame) -> dict[str, dict[str, float]]:
        numeric = season_df[list(self.features)].apply(pd.to_numeric, errors="coerce")
        minutes = pd.to_numeric(season_df["minutes"], errors="coerce").fillna(0).clip(lower=0)
        statistics: dict[str, dict[str, float]] = {}

        for feature in self.features:
            values = numeric[feature]
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median).astype("float64")
            valid_weight = minutes.where(values.notna(), 0.0)
            if feature in self.shrinkage_features and valid_weight.sum() > 0:
                prior_mean = float(np.average(filled, weights=valid_weight))
                reliability = minutes / (minutes + float(self.prior_strength))
                adjusted = reliability * filled + (1.0 - reliability) * prior_mean
            else:
                prior_mean = median
                adjusted = filled

            center = float(adjusted.mean())
            scale = float(adjusted.std(ddof=0))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            statistics[feature] = {
                "median": median,
                "prior_mean": prior_mean,
                "center": center,
                "scale": scale,
            }
        return statistics

    def fit(self, df: pd.DataFrame) -> SeasonFeaturePreprocessor:
        """Learn imputation, shrinkage priors, and scaling from complete season pools."""
        self._validate(df)
        self.season_statistics = {
            str(season): self._calculate_season_statistics(season_df)
            for season, season_df in df.groupby("season", sort=True, dropna=False)
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply persisted season transforms without fitting on a request candidate subset."""
        self._validate(df)
        result = df.copy()
        if not self.season_statistics:
            raise ValueError("SeasonFeaturePreprocessor must be fitted before transform().")

        for season, index in result.groupby("season", sort=False, dropna=False).groups.items():
            season_key = str(season)
            stats = self.season_statistics.get(season_key)
            if stats is None:
                # A newly ingested season is still normalized against its complete pool,
                # never the candidates retained by one recommendation request.
                stats = self._calculate_season_statistics(result.loc[index])

            minutes = pd.to_numeric(result.loc[index, "minutes"], errors="coerce").fillna(0).clip(lower=0)
            reliability = minutes / (minutes + float(self.prior_strength))
            for feature in self.features:
                feature_stats = stats[feature]
                values = pd.to_numeric(result.loc[index, feature], errors="coerce").fillna(
                    feature_stats["median"]
                )
                if feature in self.shrinkage_features:
                    values = reliability * values + (1.0 - reliability) * feature_stats["prior_mean"]
                result.loc[index, normalized_feature_name(feature)] = (
                    values - feature_stats["center"]
                ) / feature_stats["scale"]

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit season statistics and return transformed player profiles."""
        return self.fit(df).transform(df)


def build_pair_features(
    target: pd.Series,
    candidates: pd.DataFrame,
    features: tuple[str, ...] = RECOMMENDATION_FEATURES,
) -> pd.DataFrame:
    """Return target-candidate absolute differences in normalized observed features."""
    output: dict[str, np.ndarray] = {}
    for feature in features:
        normalized = normalized_feature_name(feature)
        if normalized not in candidates.columns or normalized not in target.index:
            raise KeyError(f"Missing normalized recommendation feature: {normalized}")
        candidate_values = pd.to_numeric(candidates[normalized], errors="coerce").to_numpy(dtype="float64")
        output[f"abs_diff__{feature}"] = np.abs(candidate_values - float(target[normalized]))
    return pd.DataFrame(output, index=candidates.index)


def euclidean_similarity(pair_features: pd.DataFrame) -> np.ndarray:
    """Convert mean standardized Euclidean distance into a bounded similarity score."""
    distance = np.sqrt(np.square(pair_features.to_numpy(dtype="float64")).mean(axis=1))
    return 1.0 / (1.0 + distance)


def fit_ledoit_wolf_precision(
    normalized_profiles: pd.DataFrame,
    features: tuple[str, ...] = RECOMMENDATION_FEATURES,
) -> np.ndarray:
    """Fit a shrinkage covariance precision matrix on normalized training profiles."""
    columns = [normalized_feature_name(feature) for feature in features]
    matrix = normalized_profiles[columns].to_numpy(dtype="float64")
    return LedoitWolf().fit(matrix).precision_.astype("float64")


def mahalanobis_similarity(pair_features: pd.DataFrame, precision: np.ndarray) -> np.ndarray:
    """Convert Ledoit-Wolf Mahalanobis distances into bounded similarity scores."""
    differences = pair_features.to_numpy(dtype="float64")
    squared_distance = np.einsum("ij,jk,ik->i", differences, precision, differences)
    distance = np.sqrt(np.maximum(squared_distance, 0.0) / differences.shape[1])
    return 1.0 / (1.0 + distance)


@dataclass
class RecommendationRankerArtifact:
    """Persist the selected playing-profile ranker and its exact feature contract."""

    algorithm: str
    preprocessor: SeasonFeaturePreprocessor
    model: Any | None = None
    model_bytes: bytes | None = None
    precision: np.ndarray | None = None
    version: str = RECOMMENDATION_RANKER_VERSION
    raw_features: tuple[str, ...] = RECOMMENDATION_FEATURES
    model_features: tuple[str, ...] = PAIR_FEATURES
    split_policy: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def materialize_model(self) -> None:
        """Restore the native LambdaMART model before request worker threads start."""
        if self.algorithm != "lambdamart" or self.model is not None:
            return
        if self.model_bytes is None:
            raise ValueError("LambdaMART artifact does not contain serialized model bytes.")

        from xgboost import XGBRanker

        self.model = XGBRanker(n_jobs=1)
        self.model.load_model(bytearray(self.model_bytes))
        self.model.set_params(n_jobs=1)

    def score(self, target: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
        """Score one complete eligible candidate pool with the selected algorithm."""
        pair_features = build_pair_features(target, candidates, self.raw_features)
        pair_features = pair_features[list(self.model_features)]
        if self.algorithm == "lambdamart":
            self.materialize_model()
            if self.model is None:
                raise ValueError("LambdaMART artifact does not contain a fitted model.")
            raw_score = np.asarray(self.model.predict(pair_features), dtype="float64")
            return 1.0 / (1.0 + np.exp(-np.clip(raw_score, -30.0, 30.0)))
        if self.algorithm == "mahalanobis":
            if self.precision is None:
                raise ValueError("Mahalanobis artifact does not contain a precision matrix.")
            return mahalanobis_similarity(pair_features, self.precision)
        if self.algorithm == "season_normalized_euclidean":
            return euclidean_similarity(pair_features)
        raise ValueError(f"Unsupported recommendation ranker algorithm: {self.algorithm}")


def save_recommendation_ranker_artifact(
    artifact: RecommendationRankerArtifact,
    path: str | Path,
) -> Path:
    """Persist one versioned recommendation ranker artifact with joblib."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    persisted = artifact
    if artifact.algorithm == "lambdamart" and artifact.model is not None:
        model_bytes = bytes(artifact.model.get_booster().save_raw(raw_format="ubj"))
        persisted = replace(artifact, model=None, model_bytes=model_bytes)
    joblib.dump(persisted, output_path)
    return output_path


def load_recommendation_ranker_artifact(
    path: str | Path,
    required: bool = False,
) -> RecommendationRankerArtifact | None:
    """Load a recommendation ranker, optionally falling back when no artifact exists."""
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        if required:
            raise FileNotFoundError(f"Recommendation ranker artifact not found: {artifact_path}")
        return None
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, RecommendationRankerArtifact):
        raise TypeError(f"Unexpected recommendation artifact type: {type(artifact)!r}")
    artifact.materialize_model()
    return artifact
