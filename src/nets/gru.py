import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(
        self,
        input_size: int,    # Size of the vocabulary (input dimension)
        embedding_dim: int, # Embedding size (dimension of word embeddings)
        hidden_size: int,   # Number of units in the GRU layer (hidden size)
        num_layers: int,    # Number of stacked GRU layers
        num_classes: int,   # Number of output classes (languages)
        dropout_rate: float # Dropout rate for regularization
    ) -> None:
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        # Embedding layer to convert input tokens to embeddings
        self.embedding = nn.Embedding(input_size, embedding_dim)

        # GRU layer
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        # Fully connected layer to map GRU output to class predictions
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional GRU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input tensor is of type float32 (necessary for GRU)
        x = x.long()  # Input tokens should be of type long, corresponding to indices

        # Pass the input through the embedding layer
        embedded = self.embedding(x)  # Shape will be (batch_size, sequence_length, embedding_dim)

        # GRU forward pass
        gru_out, _ = self.gru(embedded)

        # Use the output from the last time step (or last time step of the bidirectional GRU)
        out = gru_out[:, -1, :]  # Taking the output of the last timestep

        # Pass through the fully connected layer
        out = self.fc(out)
        return out
