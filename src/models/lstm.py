from __future__ import annotations

import torch
from torch import nn


class ShortTermLSTM(nn.Module):
    """LSTM with static context fusion for short-term NBA production forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        static_size: int,
        static_hidden_size: int = 16,
        fusion_hidden_size: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.sequence_dropout = nn.Dropout(dropout)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_size, static_hidden_size),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_hidden_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, 1),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted delta from sequence and static inputs."""
        _, (hidden, _) = self.lstm(x_seq)
        h_last = hidden[-1]
        h_last = self.sequence_dropout(h_last)
        static_context = self.static_encoder(x_static)
        combined = torch.cat([h_last, static_context], dim=1)
        prediction = self.fc(combined)
        return prediction.squeeze(1)
