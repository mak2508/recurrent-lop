#
# Adapted from:
# https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/algos/bp.py
#

import torch
import torch.nn.functional as F
from torch import optim

from .algo_type import AlgoType

class Backprop(AlgoType):
    def __init__(
        self,
        net: torch.nn.Module,
        step_size: float = 0.001,
        loss: str = 'nll',
        opt: str = 'adam',
        weight_decay: float = 0.0,
        to_perturb: bool = False,
        perturb_scale: float = 0.1,
        device: str = 'cpu',
    ) -> None:
        self.net = net
        self.to_perturb = to_perturb
        self.perturb_scale = perturb_scale
        self.device = device

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, weight_decay=weight_decay)

        # define the loss function
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]

    def learn(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Learn using one step of gradient-descent
        :param x: input
        :param target: desired output
        :return: loss
        """
        self.opt.zero_grad()
        output = self.net.forward(x=x)
        loss = self.loss_func(output, target)

        loss.backward()
        self.opt.step()

        if self.to_perturb:
            self._perturb()
        if self.loss == 'nll':
            return loss.detach(), output.detach()
        return loss.detach()

    def _perturb(self):
        with torch.no_grad():
            for i in range(int(len(self.net.layers)/2)+1):
                self.net.layers[i * 2].bias +=\
                    torch.empty(self.net.layers[i * 2].bias.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
                self.net.layers[i * 2].weight +=\
                    torch.empty(self.net.layers[i * 2].weight.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
