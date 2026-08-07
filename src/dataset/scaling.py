from __future__ import annotations

import numpy as np

from sklearn.preprocessing import StandardScaler


def scale_lstm_inputs(
    X_seq: np.ndarray,
    X_static: np.ndarray,
    y_delta: np.ndarray,
    train_mask: np.ndarray,
    scale_target_delta: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    StandardScaler,
    StandardScaler | None,
]:
    """
    Scale LSTM inputs using statistics from training samples only.

    Args:
        X_seq:
            Sequence input.

            Shape:
                (samples, sequence_length, sequence_features)

        X_static:
            Static player features.

            Shape:
                (samples, static_features)

        y_delta:
            Delta target.

            Shape:
                (samples,)

        train_mask:
            Boolean mask indicating training samples.

            Example:
                split == "train"

        scale_target_delta:
            Whether to standardize target delta.

    Returns:
        X_seq_scaled:
            Scaled sequence features.

        X_static_scaled:
            Scaled static features.

        y_delta_model:
            Scaled or original target used for training.

        seq_scaler:
            Fitted scaler for sequence features.

        static_scaler:
            Fitted scaler for static features.

        y_scaler:
            Fitted target scaler if target scaling enabled.
    """



    seq_scaler = StandardScaler()

    n_sequence_features = X_seq.shape[-1]


    # Convert:
    # (samples, seq_len, features)
    #
    # into:
    # (samples * seq_len, features)
    #
    # because StandardScaler expects 2D input

    X_seq_train = (
        X_seq[train_mask]
        .reshape(-1, n_sequence_features)
    )


    seq_scaler.fit(
        X_seq_train
    )


    X_seq_scaled = (
        seq_scaler
        .transform(
            X_seq.reshape(
                -1,
                n_sequence_features
            )
        )
        .reshape(
            X_seq.shape
        )
        .astype("float32")
    )


    static_scaler = StandardScaler()


    static_scaler.fit(
        X_static[train_mask]
    )


    X_static_scaled = (
        static_scaler
        .transform(
            X_static
        )
        .astype("float32")
    )


    if scale_target_delta:

        y_scaler = StandardScaler()


        y_scaler.fit(
            y_delta[train_mask]
            .reshape(-1, 1)
        )


        y_delta_model = (
            y_scaler
            .transform(
                y_delta.reshape(-1, 1)
            )
            .reshape(-1)
            .astype("float32")
        )

    else:

        y_scaler = None

        y_delta_model = (
            y_delta
            .astype("float32")
        )


    return (
        X_seq_scaled,
        X_static_scaled,
        y_delta_model,
        seq_scaler,
        static_scaler,
        y_scaler,
    )