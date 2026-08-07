from __future__ import annotations
import pandas as pd
import numpy as np


def prepare_sequence_training(df:pd.DataFrame, task_config: dict) -> pd.DataFrame:
    sequence_df = df.copy()
    required_columns = [
        "player_id",
        "season",
        "as_of_date",
        "game_id",
        "split",
        "min",
        "min_season_avg",

        task_config["stat_col"],
        task_config["stat_avg_col"],
        task_config["target_col"],
    ]
    missing = set(required_columns) - set(sequence_df.columns())
    if missing:
        raise KeyError(f"Missing columns: {missing}") 
    before = len(sequence_df)
    sequence_df = (sequence_df.drop_na(subset=required_columns)).copy()
    sequence_df = sequence_df.sort_values(["player_id", "season", "as_of_date", "game_id"]).reset_index(drop=True)
    print(
        f"Dropped rows: {before:,} -> {len(sequence_df):,}"
    )


    return sequence_df



def make_lstm_delta_sequences(
    df: pd.DataFrame,
    task_config: dict[str, object],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    stat_col = str(task_config["stat_col"])
    stat_avg_col = str(task_config["stat_avg_col"])
    target_col = str(task_config["target_col"])
    sequence_length = int(task_config["sequence_length"])

    # Store generated samples

    rows_x_seq = []
    rows_x_static = []

    rows_y_delta = []
    rows_y_actual = []

    rows_baseline = []
    rows_split = []

    ordered_df = (df.sort_values(subset=["player_id", "season", "as_of_date", "game_id"])).copy()
    for _, group in ordered_df.groupby(
        [
            "player_id",
            "season",
        ],
        sort=False,
    ):
        stat_delta = group[stat_col].to_numpy(dtype="float32") - group[stat_avg_col].to_numpy(dtype="float32")
        min_delta = group["min"].to_numpy(dtype="float32") - group["min_season_avg"].to_numpy("float32")

        sequence_values = np.column_stack(
            [
                stat_delta,
                min_delta,
            ]
        ).astype("float32")

        # (num_games, 2)
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

        #sliding window construction
        for idx in range(
            sequence_length - 1,
            len(group)
        ):
            window = (sequence_values[idx - sequence_length + 1 : idx + 1])
        # Current player context
            static_context = static_values[idx]


            # Future target
            target_actual = targets_actual[idx]


            # Baseline used for delta prediction
            baseline = baselines[idx]


            # Residual target
            target_delta = (
                target_actual
                -
                baseline
            )


            # Safety check
            if (
                np.isnan(window).any()
                or np.isnan(static_context).any()
                or np.isnan(target_actual)
                or np.isnan(target_delta)
            ):
                continue

            rows_x_seq.append(window)

            rows_x_static.append(
                static_context
            )

            rows_y_delta.append(
                target_delta
            )

            rows_y_actual.append(
                target_actual
            )

            rows_baseline.append(
                baseline
            )

            rows_split.append(
                splits[idx]
            )
    return (
        np.asarray(
            rows_x_seq,
            dtype="float32",
        ),

        np.asarray(
            rows_x_static,
            dtype="float32",
        ),

        np.asarray(
            rows_y_delta,
            dtype="float32",
        ),

        np.asarray(
            rows_y_actual,
            dtype="float32",
        ),

        np.asarray(
            rows_baseline,
            dtype="float32",
        ),

        np.asarray(
            rows_split,
        ),
    )
        