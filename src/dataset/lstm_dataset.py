from __future__ import annotations

import torch

from torch.utils.data import Dataset

import numpy as np


class ShortTermLSTMDataset(Dataset):

    def __init__(
        self,
        X_seq: np.ndarray,
        X_static: np.ndarray,
        y_delta: np.ndarray,
        y_actual: np.ndarray | None = None,
        baseline: np.ndarray | None = None,
        split: np.ndarray | None = None,
    ):
        """
        Dataset for short-term NBA LSTM prediction.

        X_seq:
            (samples, sequence_length, sequence_features)

        X_static:
            (samples, static_features)

        y_delta:
            residual target used for training

        y_actual:
            original target for evaluation

        baseline:
            season average used to reconstruct prediction

        split:
            train/val/test labels
        """


        self.X_seq = torch.tensor(
            X_seq,
            dtype=torch.float32
        )


        self.X_static = torch.tensor(
            X_static,
            dtype=torch.float32
        )


        self.y_delta = torch.tensor(
            y_delta,
            dtype=torch.float32
        )


        # Keep numpy because evaluation may need original values

        self.y_actual = y_actual

        self.baseline = baseline

        self.split = split



    def __len__(self):
        return len(self.y_delta)



    def __getitem__(self, idx):

        return (
            self.X_seq[idx],
            self.X_static[idx],
            self.y_delta[idx],
        )