import torch.nn as nn

from src.algos import AlgoType, BP
from src.config import Config

def load_algo(model: nn.Module, config: Config) -> AlgoType:
    if config.algo == "BP":
        return BP(model, learning_rate=config.learning_rate, to_perturb=config.to_perturb, device=str(config.device))
    elif config.algo == "CBP":
        raise ValueError("CBP not yet implemented")
    else:
        raise ValueError(f"Unknown algorithm: {config.algo}")