from __future__ import annotations

import pandas as pd
import pytest

from config.long_term_config import resolve_long_term_task_config
from dataset.long_term_modeling import infer_long_term_feature_columns, prepare_long_term_training, validate_long_term_split


def sample_long_term_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "anchor_season": ["2019-20", "2020-21", "2021-22"],
            "anchor_season_start_year": [2019, 2020, 2021],
            "split": ["train", "validation", "test"],
            "age_at_anchor": [25.0, 26.0, 27.0],
            "pts_per_36_lag_0": [20.0, 21.0, 22.0],
            "pts_per_36_h1": [21.0, 22.0, 23.0],
        }
    )


def test_prepare_long_term_training_validates_and_sorts_rows():
    task_config = resolve_long_term_task_config("pts_per_36", 1)
    df = sample_long_term_frame().sample(frac=1.0, random_state=42)

    prepared = prepare_long_term_training(
        df,
        task_config=task_config,
        feature_cols=["age_at_anchor", "pts_per_36_lag_0"],
    )

    assert prepared["anchor_season_start_year"].tolist() == [2019, 2020, 2021]
    assert prepared["player_id"].tolist() == [1, 1, 1]


def test_infer_long_term_feature_columns_excludes_future_targets_and_metadata():
    df = pd.DataFrame(
        {
            "player_id": [1],
            "player_name": ["Sample"],
            "anchor_season": ["2021-22"],
            "split": ["test"],
            "age_at_anchor": [25.0],
            "pts_per_36_lag_0": [20.0],
            "active_h1": [1],
            "pts_per_36_h1": [21.0],
            "age_at_h1": [26.0],
        }
    )

    features = infer_long_term_feature_columns(df)

    assert features == ["age_at_anchor", "pts_per_36_lag_0"]


def test_prepare_long_term_training_keeps_missing_features_for_model_preprocessor():
    task_config = resolve_long_term_task_config("pts_per_36", 1)
    df = sample_long_term_frame()
    df.loc[df["anchor_season"].eq("2020-21"), "pts_per_36_lag_0"] = None

    prepared = prepare_long_term_training(
        df,
        task_config=task_config,
        feature_cols=["age_at_anchor", "pts_per_36_lag_0"],
    )

    assert len(prepared) == 3
    assert prepared["pts_per_36_lag_0"].isna().sum() == 1


def test_prepare_long_term_training_reports_missing_columns():
    task_config = resolve_long_term_task_config("pts_per_36", 1)
    df = sample_long_term_frame().drop(columns=["pts_per_36_lag_0"])

    with pytest.raises(KeyError, match="Missing long-term columns"):
        prepare_long_term_training(
            df,
            task_config=task_config,
            feature_cols=["age_at_anchor", "pts_per_36_lag_0"],
        )


def test_validate_long_term_split_rejects_mismatched_season_map():
    df = sample_long_term_frame()
    df.loc[df["anchor_season"].eq("2020-21"), "split"] = "train"

    with pytest.raises(ValueError, match="does not match configured season map"):
        validate_long_term_split(df)
