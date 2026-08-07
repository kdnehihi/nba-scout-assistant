from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT)
)
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config.lstm_config import LSTM_TASK_CONFIG

from src.dataset.sequence import (
    prepare_sequence_training,
    make_lstm_delta_sequences,
)

from src.dataset.loaders import (
    DataPaths,
    load_performance_training_clean,
    load_player_game_logs,
)

from src.dataset.scaling import (
    scale_lstm_inputs,
)

from src.dataset.lstm_dataset import (
    ShortTermLSTMDataset,
)

from src.models.lstm import (
    ShortTermLSTM,
)


# =====================================================
# Training config
# =====================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


TASK = "pts"

BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 1e-3


ARTIFACT_DIR = Path(
    "artifacts"
)

MODEL_PATH = (
    ARTIFACT_DIR
    /
    "short_term_lstm_pts.pt"
)


# =====================================================
# Data loading
# =====================================================


def load_short_term_dataframe():
    """
    Load gold point-in-time performance table.

    Replace this with your real data access layer.

    Expected columns:

    player_id
    season
    as_of_date
    game_id

    pts
    pts_season_avg

    min
    min_season_avg

    target_next_5_pts

    split
    """

    path = DataPaths()
    df = load_performance_training_clean(path)
    return df



# =====================================================
# Train one epoch
# =====================================================


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    total_loss = 0.0


    for batch in loader:

        x_seq, x_static, y = batch


        x_seq = x_seq.to(DEVICE)
        x_static = x_static.to(DEVICE)
        y = y.to(DEVICE)


        optimizer.zero_grad()


        prediction = model(
            x_seq,
            x_static,
        )


        loss = criterion(
            prediction,
            y,
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()


    return total_loss / len(loader)



# =====================================================
# Validation
# =====================================================


def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0


    with torch.no_grad():

        for batch in loader:

            x_seq, x_static, y = batch


            x_seq = x_seq.to(DEVICE)
            x_static = x_static.to(DEVICE)
            y = y.to(DEVICE)


            prediction = model(
                x_seq,
                x_static,
            )


            loss = criterion(
                prediction,
                y,
            )


            total_loss += loss.item()


    return total_loss / len(loader)



# =====================================================
# Main
# =====================================================


def train_short_term():

    print(
        f"Using device: {DEVICE}"
    )


    # ------------------------------
    # 1. Load dataframe
    # ------------------------------

    df = load_short_term_dataframe()



    task_config = (
        LSTM_TASK_CONFIG[TASK]
    )



    # ------------------------------
    # 2. Prepare sequence table
    # ------------------------------

    sequence_df = (
        prepare_sequence_training(
            df,
            task_config,
        )
    )



    # ------------------------------
    # 3. Build sequences
    # ------------------------------

    (
        X_seq,
        X_static,
        y_delta,
        y_actual,
        baseline,
        split,
    ) = make_lstm_delta_sequences(
        sequence_df,
        task_config,
    )



    train_mask = (
        split == "train"
    )


    val_mask = (
        split == "validation"
    )



    # ------------------------------
    # 4. Scaling
    # ------------------------------

    (
        X_seq_scaled,
        X_static_scaled,
        y_delta_scaled,
        seq_scaler,
        static_scaler,
        y_scaler,
    ) = scale_lstm_inputs(
        X_seq,
        X_static,
        y_delta,
        train_mask,
        scale_target_delta=True,
    )



    # ------------------------------
    # 5. Dataset
    # ------------------------------

    train_dataset = ShortTermLSTMDataset(
        X_seq_scaled[train_mask],
        X_static_scaled[train_mask],
        y_delta_scaled[train_mask],
        y_actual[train_mask],
        baseline[train_mask],
        split[train_mask],
    )


    val_dataset = ShortTermLSTMDataset(
        X_seq_scaled[val_mask],
        X_static_scaled[val_mask],
        y_delta_scaled[val_mask],
        y_actual[val_mask],
        baseline[val_mask],
        split[val_mask],
    )



    # ------------------------------
    # 6. DataLoader
    # ------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )



    # ------------------------------
    # 7. Model
    # ------------------------------

    model = ShortTermLSTM(
        input_size=X_seq.shape[-1],
        hidden_size=48,
        static_size=X_static.shape[-1],
    )


    model.to(DEVICE)



    criterion = nn.MSELoss()


    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )



    # ------------------------------
    # 8. Training loop
    # ------------------------------

    best_val_loss = float("inf")


    for epoch in range(EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )


        val_loss = evaluate(
            model,
            val_loader,
            criterion,
        )


        print(
            f"""
Epoch {epoch + 1}/{EPOCHS}

Train Loss:
{train_loss:.5f}

Val Loss:
{val_loss:.5f}
"""
        )


        if val_loss < best_val_loss:

            best_val_loss = val_loss


            ARTIFACT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )


            torch.save(
                {
                    "model_state_dict": model.state_dict(),

                    "task": TASK,

                    "best_val_loss": best_val_loss,

                    "seq_scaler": seq_scaler,

                    "static_scaler": static_scaler,

                    "y_scaler": y_scaler,
                },
                MODEL_PATH,
            )


            print(
                "Saved best model"
            )



if __name__ == "__main__":

    train_short_term()