from abc import ABC, abstractmethod
import torch
from typing import Union, Tuple

class AlgoType(ABC):
    @abstractmethod
    def learn(self, x: torch.Tensor, target: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Learn using one step of the algorithm
        :param x: input
        :param target: desired output
        :return: loss or (loss, output)
        """
        pass