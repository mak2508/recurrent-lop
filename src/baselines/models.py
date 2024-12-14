import torch.nn as nn


# Define the MLP model
class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        dropout_rate: float,
    ) -> None:
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float,
    ) -> None:
        super(LSTM, self).__init__()

        # MNIST images are 28x28, we'll treat each row as a sequence of length 28 with 28 features
        self.sequence_length = 28
        self.input_size = 28  # number of features per timestep

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input: (batch_size, channels, height, width) -> (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        x = x.view(batch_size, self.sequence_length, self.input_size)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Use the output from the last time step
        out = lstm_out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(out)
        return out
