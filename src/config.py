from dataclasses import dataclass
from typing import List, Literal
import torch

@dataclass
class Config:
    # Model parameters
    input_size: int = 10000  # Vocabulary size (size of the word index set)
    embedding_dim: int = 64  # Embedding dimension (word vector size)
    hidden_size: int = 64    # Number of units in the GRU layer (hidden size)
    num_layers: int = 1      # Number of stacked GRU layers
    num_classes: int = 10    # Number of output classes (languages)
    dropout_rate: float = 0.0
    model_type: Literal["GRU", "MLP", "LSTM"] = "GRU"  # Type of model: GRU, LSTM, or MLP
    vocab_limit: int = 10000  # Limit of words in vocabulary
    max_length: int = 50      # Maximum sequence length (words per sentence)

    # Training parameters
    num_epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.01
    num_tasks: int = 3        # Number of tasks to repeat
    algo: Literal["BP", "CBP_MLP, CBP_LSTM"] = "BP"
    to_perturb: bool = False
    perturb_scale: float = 0.1  # Scale for perturbation during training
    loss: Literal["nll", "mse"] = "nll"  # Loss function type
    opt: Literal["adam", "sgd"] = "adam"  # Optimizer type
    weight_decay: float = 0.0  # Weight decay for optimizer

    # Language Dataset parameters
    languages: List[str] = ("spa", "por", "ita", "fra", "ron", "deu", "cmn", "rus", "hin", "ara")
    train_sentences_per_class: int = 900
    test_sentences_per_class: int = 100
    sentences_per_class: int = 1000  # Combined train + test

    # Experiment description
    exp_desc: str = "gru_language_baseline"
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        return cls(
            input_size=config_dict.get('model', {}).get('input_size', cls.input_size),
            embedding_dim=config_dict.get('model', {}).get('embedding_dim', cls.embedding_dim),
            hidden_size=config_dict.get('model', {}).get('hidden_size', cls.hidden_size),
            num_layers=config_dict.get('model', {}).get('num_layers', cls.num_layers),
            num_classes=config_dict.get('model', {}).get('num_classes', cls.num_classes),
            dropout_rate=config_dict.get('model', {}).get('dropout_rate', cls.dropout_rate),
            model_type=config_dict.get('model', {}).get('model_type', cls.model_type),
            vocab_limit=config_dict.get('model', {}).get('vocab_limit', cls.vocab_limit),
            max_length=config_dict.get('model', {}).get('max_length', cls.max_length),
            num_epochs=config_dict.get('training', {}).get('num_epochs', cls.num_epochs),
            batch_size=config_dict.get('training', {}).get('batch_size', cls.batch_size),
            learning_rate=config_dict.get('training', {}).get('learning_rate', cls.learning_rate),
            num_tasks=config_dict.get('training', {}).get('num_tasks', cls.num_tasks),
            algo=config_dict.get('training', {}).get('algo', cls.algo),
            to_perturb=config_dict.get('training', {}).get('to_perturb', cls.to_perturb),
            perturb_scale=config_dict.get('training', {}).get('perturb_scale', cls.perturb_scale),
            loss=config_dict.get('training', {}).get('loss', cls.loss),
            opt=config_dict.get('training', {}).get('opt', cls.opt),
            weight_decay=config_dict.get('training', {}).get('weight_decay', cls.weight_decay),
            languages=config_dict.get('languages', cls.languages),
            train_sentences_per_class=config_dict.get('train_sentences_per_class', cls.train_sentences_per_class),
            test_sentences_per_class=config_dict.get('test_sentences_per_class', cls.test_sentences_per_class),
            sentences_per_class=config_dict.get('sentences_per_class', cls.sentences_per_class),
            exp_desc=config_dict.get('exp_desc', cls.exp_desc),
            device=torch.device(config_dict.get('device', cls.device)),
        )