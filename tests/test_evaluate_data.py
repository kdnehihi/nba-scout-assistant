from __future__ import annotations

import pytest
import pandas as pd

from evaluation.evaluate_data import (
    DataQualityConfig,
    assert_training_dataframe_is_valid,
    validate_training_dataframe,
)


def test_validate_training_dataframe_passes_clean_data():
    df = pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "game_id": [10, 11, 12],
            "split": ["train", "validation", "test"],
            "feature_a": [1.0, 2.0, 3.0],
            "target": [4.0, 5.0, 6.0],
        }
    )
    config = DataQualityConfig(
        required_columns=("player_id", "game_id"),
        feature_columns=("feature_a",),
        target_columns=("target",),
        key_columns=("player_id", "game_id"),
    )

    issues, audit = validate_training_dataframe(df, config)

    assert issues.empty
    assert audit["split_summary"]["rows"].sum() == 3


def test_validate_training_dataframe_flags_missing_target_and_split():
    df = pd.DataFrame(
        {
            "player_id": [1, 1],
            "game_id": [10, 11],
            "split": ["train", "future"],
            "feature_a": [1.0, 2.0],
            "target": [4.0, None],
        }
    )
    config = DataQualityConfig(
        required_columns=("player_id", "game_id"),
        feature_columns=("feature_a",),
        target_columns=("target",),
        allowed_splits=("train", "validation", "test"),
        require_all_splits=True,
    )

    issues, _ = validate_training_dataframe(df, config)

    assert {"split_label", "split_presence", "target_missing"}.issubset(set(issues["check"]))
    with pytest.raises(ValueError):
        assert_training_dataframe_is_valid(df, config)

