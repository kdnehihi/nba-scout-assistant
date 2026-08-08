import torch
import torch.nn as nn

class ShortTermLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            static_size,
            dropout=0.15,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(
                hidden_size + static_size,
                24
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                24,
                1
            )
        )

    def forward(
        self,
        x_seq,
        x_static,
    ):

        output, (hidden, cell) = self.lstm(
            x_seq
        )

        # hidden:
        # (num_layers, batch, hidden_size)

        h_last = hidden[-1]


        combined = torch.cat(
            [
                h_last,
                x_static
            ],
            dim=1
        )


        prediction = self.fc(
            combined
        )

        return prediction.squeeze(1)