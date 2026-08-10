from __future__ import annotations

import torch
from torch import nn


class LongTermMLP(nn.Module):
    """Tabular MLP used for long-term player forecasting tasks."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int = 1,
        dropout: float = 0.10,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = hidden_size

        layers.append(nn.Linear(current_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one prediction value or logit per input row."""
        prediction = self.network(x)
        return prediction.squeeze(1)
