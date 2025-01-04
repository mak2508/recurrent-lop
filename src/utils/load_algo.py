import torch.nn as nn

from src.algos import AlgoType, BP
from src.config import Config

def load_algo(model: nn.Module, config: Config) -> AlgoType:
    """
    Load the specified algorithm using all parameters from the configuration.

    Args:
        model (nn.Module): The neural network model.
        config (Config): Configuration object containing algorithm parameters.

    Returns:
        AlgoType: An instance of the specified algorithm.
    """
    if config.algo == "BP":
        return BP(
            net=model,
            learning_rate=config.learning_rate,
            loss=config.loss,
            opt=config.opt,
            weight_decay=config.weight_decay,
            to_perturb=config.to_perturb,
            perturb_scale=config.perturb_scale,
            device=str(config.device)
        )
    elif config.algo == "CBP":
        raise ValueError("CBP not yet implemented")
    else:
        raise ValueError(f"Unknown algorithm: {config.algo}")