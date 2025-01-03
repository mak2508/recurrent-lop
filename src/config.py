from dataclasses import dataclass
from typing import Literal
import torch

@dataclass
class Config:
    # Model parameters
    input_size: int = 784
    hidden_size: int = 25
    num_classes: int = 10
    dropout_rate: float = 0.0
    model_type: Literal["MLP", "LSTM"] = "MLP"

    # Training parameters
    algo: Literal["BP", "CBP"] = "BP"
    num_epochs: int = 2
    batch_size: int = 6000
    learning_rate: float = 0.01
    num_shuffles: int = 3
    to_perturb: bool = False
    # Experiment metadata
    exp_desc: str = "mnist_reshuffle"
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        return cls(
            hidden_size=config_dict.get('model', {}).get('hidden_size', cls.hidden_size),
            dropout_rate=config_dict.get('model', {}).get('dropout_rate', cls.dropout_rate),
            model_type=config_dict.get('model', {}).get('model_type', cls.model_type),
            algo=config_dict.get('training', {}).get('algo', cls.algo),
            num_epochs=config_dict.get('training', {}).get('num_epochs', cls.num_epochs),
            batch_size=config_dict.get('training', {}).get('batch_size', cls.batch_size),
            learning_rate=config_dict.get('training', {}).get('learning_rate', cls.learning_rate),
            num_shuffles=config_dict.get('training', {}).get('num_shuffles', cls.num_shuffles),
            to_perturb=config_dict.get('training', {}).get('to_perturb', cls.to_perturb),
            exp_desc=config_dict.get('exp_desc', cls.exp_desc)
        )