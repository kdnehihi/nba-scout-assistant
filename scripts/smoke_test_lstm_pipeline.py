"""
Smoke test for NBA LSTM pipeline.

Run from project root:
    python scripts/smoke_test_lstm_pipeline.py

This creates synthetic raw-like NBA data and validates:
raw dataframe
-> sequence preparation
-> sequence building
-> scaling
-> Dataset
-> DataLoader
-> LSTM forward
-> backward pass
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# Allow running from project root
ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(
    0,
    str(ROOT)
)


from src.config.lstm_config import LSTM_TASK_CONFIG
from src.dataset.sequence import (
    prepare_sequence_training,
    make_lstm_delta_sequences,
)
from src.dataset.scaling import scale_lstm_inputs
from src.dataset.lstm_dataset import ShortTermLSTMDataset
from src.models.lstm import ShortTermLSTM


def create_fake_raw_nba_data(
    n_players=3,
    n_games=40,
):
    rows = []

    players = [
        f"player_{i}"
        for i in range(n_players)
    ]

    for player_id in players:
        season_avg_pts = np.random.uniform(15, 30)
        season_avg_min = np.random.uniform(25, 36)

        for game_id in range(n_games):
            pts = season_avg_pts + np.random.normal(0, 5)
            minutes = season_avg_min + np.random.normal(0, 4)

            rows.append(
                {
                    "player_id": player_id,
                    "season": 2024,
                    "as_of_date": pd.Timestamp("2024-01-01")
                    + pd.Timedelta(days=game_id),
                    "game_id": game_id,
                    "pts": pts,
                    "pts_season_avg": season_avg_pts,
                    "min": minutes,
                    "min_season_avg": season_avg_min,
                    # Fake future target
                    "target_next_5_pts_avg": pts + np.random.normal(0, 3),
                    "split": "train" if game_id < 30 else "test",
                }
            )

    return pd.DataFrame(rows)


def main():

    print("\n1. Creating fake raw NBA data")
    raw_df = create_fake_raw_nba_data()

    print(raw_df.head())
    print(raw_df.shape)


    config = LSTM_TASK_CONFIG["points"]


    print("\n2. Prepare sequence dataframe")
    sequence_df = prepare_sequence_training(
        raw_df,
        config,
    )

    print(sequence_df.shape)


    print("\n3. Build LSTM sequences")

    (
        X_seq,
        X_static,
        y_delta,
        y_actual,
        baseline,
        split,
    ) = make_lstm_delta_sequences(
        sequence_df,
        config,
    )

    print("X_seq:", X_seq.shape)
    print("X_static:", X_static.shape)
    print("y_delta:", y_delta.shape)


    print("\n4. Scaling")

    train_mask = split == "train"

    (
        X_seq_scaled,
        X_static_scaled,
        y_delta_scaled,
        _,
        _,
        _,
    ) = scale_lstm_inputs(
        X_seq,
        X_static,
        y_delta,
        train_mask,
        scale_target_delta=True,
    )


    print(X_seq_scaled.shape)
    print(X_static_scaled.shape)


    print("\n5. Dataset + DataLoader")

    dataset = ShortTermLSTMDataset(
        X_seq_scaled,
        X_static_scaled,
        y_delta_scaled,
        y_actual,
        baseline,
        split,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    batch = next(iter(loader))

    print("sequence:", batch[0].shape)
    print("static:", batch[1].shape)
    print("target:", batch[2].shape)


    print("\n6. Model forward")

    model = ShortTermLSTM(
        input_size=2,
        hidden_size=48,
        static_size=2,
    )


    prediction = model(
        batch[0],
        batch[1],
    )

    print("prediction:", prediction.shape)


    print("\n7. Backward test")

    loss_fn = torch.nn.MSELoss()

    loss = loss_fn(
        prediction,
        batch[2],
    )

    loss.backward()

    print("loss:", loss.item())
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
