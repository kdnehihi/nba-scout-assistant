from __future__ import annotations

import pandas as pd

from evaluation.metrics import range_metrics, regression_metrics
from evaluation.evaluate_short_term import evaluate_lstm_predictions_by_split
from modeling.baselines import naive_last_5_prediction, season_average_prediction
from training.splitters import feature_target_split, split_by_column


def test_regression_and_range_metrics():
    regression = regression_metrics([1, 2, 3], [1, 2, 4])
    ranged = range_metrics([1, 2, 3], [1, 2, 4], [0, 1, 2], [2, 3, 4])

    assert regression["mae"] > 0
    assert ranged["coverage_rate"] == 1.0
    assert ranged["avg_range_width"] == 2.0


def test_splitters_and_baselines():
    df = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "pts_last_5": [10, 11, 12],
            "pts_season_avg": [9, 10, 11],
            "target": [13, 14, 15],
        }
    )
    train, validation, test = split_by_column(df)
    x_train, y_train = feature_target_split(train, ["pts_last_5"], "target")

    assert len(train) == len(validation) == len(test) == 1
    assert x_train["pts_last_5"].iloc[0] == 10
    assert y_train.iloc[0] == 13
    assert naive_last_5_prediction(df, "pts").tolist() == [10, 11, 12]
    assert season_average_prediction(df, "pts").tolist() == [9, 10, 11]


def test_short_term_lstm_evaluation_uses_restored_targets():
    metrics = evaluate_lstm_predictions_by_split(
        y_actual=pd.Series([10.0, 12.0, 20.0, 22.0]).to_numpy(),
        y_pred=pd.Series([9.0, 13.0, 18.0, 24.0]).to_numpy(),
        split=pd.Series(["validation", "validation", "test", "test"]).to_numpy(),
        task="points",
    )

    validation = metrics[metrics["split"].eq("validation")].iloc[0]
    test = metrics[metrics["split"].eq("test")].iloc[0]

    assert validation["mae"] == 1.0
    assert test["mae"] == 2.0
