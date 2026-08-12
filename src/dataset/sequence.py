from __future__ import annotations

import numpy as np
import pandas as pd


def make_lstm_delta_inference_window(
    df: pd.DataFrame,
    task_config,
) -> tuple[np.ndarray, np.ndarray, float, pd.Series]:
    # Build one latest LSTM input window without requiring future targets.
    """Return X_seq, X_static, baseline, and anchor row for one player history."""
    required_columns = [
        "player_id",
        "season",
        "as_of_date",
        "game_id",
        "min",
        "min_season_avg",
        task_config.stat_col,
        task_config.stat_avg_col,
    ]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    rows = (
        df
        .dropna(subset=required_columns)
        .sort_values(["player_id", "season", "as_of_date", "game_id"])
        .copy()
    )
    if len(rows) < task_config.sequence_length:
        raise ValueError(
            f"Need at least {task_config.sequence_length} player rows; got {len(rows)}."
        )

    window_df = rows.tail(task_config.sequence_length)
    anchor = window_df.iloc[-1]
    sequence = np.column_stack(
        [
            window_df[task_config.stat_col].to_numpy(dtype="float32")
            - window_df[task_config.stat_avg_col].to_numpy(dtype="float32"),
            window_df["min"].to_numpy(dtype="float32")
            - window_df["min_season_avg"].to_numpy(dtype="float32"),
        ]
    ).astype("float32")
    static = np.asarray(
        [[anchor[task_config.stat_avg_col], anchor["min_season_avg"]]],
        dtype="float32",
    )
    X_seq = sequence.reshape(1, task_config.sequence_length, sequence.shape[-1])
    baseline = float(anchor[task_config.stat_avg_col])
    return X_seq, static, baseline, anchor


def prepare_sequence_training(
    df: pd.DataFrame,
    task_config,
) -> pd.DataFrame:
    sequence_df = df.copy()

    required_columns = [
        "player_id",
        "season",
        "as_of_date",
        "game_id",
        "split",
        "min",
        "min_season_avg",
        task_config.stat_col,
        task_config.stat_avg_col,
        task_config.target_col,
    ]

    missing = set(required_columns) - set(sequence_df.columns)

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    before = len(sequence_df)

    sequence_df = (
        sequence_df
        .dropna(subset=required_columns)
        .copy()
    )

    sequence_df = (
        sequence_df
        .sort_values(
            [
                "player_id",
                "season",
                "as_of_date",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Dropped rows: {before:,} -> {len(sequence_df):,}"
    )

    return sequence_df


def make_lstm_delta_sequences(
    df: pd.DataFrame,
    task_config,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    stat_col = task_config.stat_col
    stat_avg_col = task_config.stat_avg_col
    target_col = task_config.target_col
    sequence_length = task_config.sequence_length

    rows_x_seq = []
    rows_x_static = []

    rows_y_delta = []
    rows_y_actual = []

    rows_baseline = []
    rows_split = []


    ordered_df = (
        df
        .sort_values(
            [
                "player_id",
                "season",
                "as_of_date",
                "game_id",
            ]
        )
        .copy()
    )


    for _, group in ordered_df.groupby(
        [
            "player_id",
            "season",
        ],
        sort=False,
    ):

        stat_delta = (
            group[stat_col].to_numpy(dtype="float32")
            -
            group[stat_avg_col].to_numpy(dtype="float32")
        )

        min_delta = (
            group["min"].to_numpy(dtype="float32")
            -
            group["min_season_avg"].to_numpy(dtype="float32")
        )


        sequence_values = np.column_stack(
            [
                stat_delta,
                min_delta,
            ]
        ).astype("float32")


        static_values = (
            group[
                [
                    stat_avg_col,
                    "min_season_avg",
                ]
            ]
            .to_numpy(dtype="float32")
        )


        targets_actual = (
            group[target_col]
            .to_numpy(dtype="float32")
        )

        baselines = (
            group[stat_avg_col]
            .to_numpy(dtype="float32")
        )

        splits = (
            group["split"]
            .to_numpy()
        )


        for idx in range(
            sequence_length - 1,
            len(group),
        ):

            window = (
                sequence_values[
                    idx - sequence_length + 1:
                    idx + 1
                ]
            )

            static_context = static_values[idx]

            target_actual = targets_actual[idx]

            baseline = baselines[idx]

            target_delta = target_actual - baseline


            if (
                np.isnan(window).any()
                or np.isnan(static_context).any()
                or np.isnan(target_actual)
                or np.isnan(target_delta)
            ):
                continue


            rows_x_seq.append(window)
            rows_x_static.append(static_context)

            rows_y_delta.append(target_delta)
            rows_y_actual.append(target_actual)

            rows_baseline.append(baseline)
            rows_split.append(splits[idx])


    return (
        np.asarray(rows_x_seq, dtype="float32"),
        np.asarray(rows_x_static, dtype="float32"),
        np.asarray(rows_y_delta, dtype="float32"),
        np.asarray(rows_y_actual, dtype="float32"),
        np.asarray(rows_baseline, dtype="float32"),
        np.asarray(rows_split),
    )
