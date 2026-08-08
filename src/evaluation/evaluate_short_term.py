from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.lstm_config import resolve_lstm_task_config
from src.dataset.loaders import load_performance_training_clean, resolve_data_paths
from src.dataset.sequence import make_lstm_delta_sequences, prepare_sequence_training
from src.evaluation.metrics import regression_metrics
from src.models.lstm import ShortTermLSTM


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def apply_saved_scalers(
    X_seq: np.ndarray,
    X_static: np.ndarray,
    checkpoint: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Scale sequence and static inputs using scalers saved during training."""
    seq_scaler = checkpoint["seq_scaler"]
    static_scaler = checkpoint["static_scaler"]

    n_sequence_features = X_seq.shape[-1]
    X_seq_scaled = (
        seq_scaler.transform(X_seq.reshape(-1, n_sequence_features))
        .reshape(X_seq.shape)
        .astype("float32")
    )
    X_static_scaled = static_scaler.transform(X_static).astype("float32")
    return X_seq_scaled, X_static_scaled


def predict_lstm_actuals(
    model: ShortTermLSTM,
    X_seq_scaled: np.ndarray,
    X_static_scaled: np.ndarray,
    baseline: np.ndarray,
    y_scaler,
    batch_size: int,
    device: str = DEVICE,
) -> np.ndarray:
    """Predict next-five-game averages by restoring model deltas to stat scale."""
    predictions: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(X_seq_scaled), batch_size):
            end = start + batch_size
            x_seq = torch.tensor(X_seq_scaled[start:end], dtype=torch.float32, device=device)
            x_static = torch.tensor(X_static_scaled[start:end], dtype=torch.float32, device=device)
            pred_delta = model(x_seq, x_static).cpu().numpy()
            predictions.append(pred_delta)

    pred_delta = np.concatenate(predictions)
    if y_scaler is not None:
        pred_delta = y_scaler.inverse_transform(pred_delta.reshape(-1, 1)).reshape(-1)
    return baseline + pred_delta


def evaluate_lstm_predictions_by_split(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    split: np.ndarray,
    task: str,
    splits: tuple[str, ...] = ("validation", "test"),
) -> pd.DataFrame:
    """Evaluate restored LSTM predictions against next-five-game actual targets."""
    rows = []
    for split_name in splits:
        mask = split == split_name
        if not mask.any():
            continue
        rows.append(
            {
                "task": task,
                "split": split_name,
                "rows": int(mask.sum()),
                **regression_metrics(y_actual[mask], y_pred[mask]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_short_term_checkpoint(
    model_path: Path | str,
    data_dir: Path | str = "data",
    task: str | None = None,
    splits: tuple[str, ...] = ("validation", "test"),
) -> pd.DataFrame:
    """Load a short-term LSTM checkpoint and return MAE/RMSE/R2 by split."""
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    task_name = task or str(checkpoint["task"])
    task_name, task_config = resolve_lstm_task_config(task_name)

    paths = resolve_data_paths(data_dir)
    performance = load_performance_training_clean(paths)
    sequence_df = prepare_sequence_training(performance, task_config)
    X_seq, X_static, _, y_actual, baseline, split = make_lstm_delta_sequences(sequence_df, task_config)
    X_seq_scaled, X_static_scaled = apply_saved_scalers(X_seq, X_static, checkpoint)

    model = ShortTermLSTM(
        input_size=X_seq.shape[-1],
        hidden_size=int(checkpoint.get("hidden_size", task_config.hidden_size)),
        static_size=X_static.shape[-1],
        dropout=float(checkpoint.get("dropout", task_config.dropout)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    rows = []
    for split_name in splits:
        mask = split == split_name
        if not mask.any():
            continue
        pred_actual = predict_lstm_actuals(
            model=model,
            X_seq_scaled=X_seq_scaled[mask],
            X_static_scaled=X_static_scaled[mask],
            baseline=baseline[mask],
            y_scaler=checkpoint.get("y_scaler"),
            batch_size=task_config.batch_size,
        )
        rows.append(
            {
                "task": task_name,
                "split": split_name,
                "rows": int(mask.sum()),
                **regression_metrics(y_actual[mask], pred_actual),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for checkpoint evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--task", default=None)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluation = evaluate_short_term_checkpoint(
        model_path=args.model_path,
        data_dir=args.data_dir,
        task=args.task,
        splits=tuple(args.splits),
    )
    print(evaluation.to_string(index=False))
