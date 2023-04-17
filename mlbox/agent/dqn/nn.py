from itertools import chain

import torch as T
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential


class DQNNet(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 device: T.device,
                 hidden_dim: int = 64,
                 hidden_n: int = 1,
                 Activation: type[Module] = ReLU):
        super().__init__()
        # net
        self.net = Sequential(
            # input
            Linear(in_dim, hidden_dim),
            Activation(),
            # hidden
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                Activation(),
            ) for _ in range(hidden_n))),
            # output
            Linear(hidden_dim, out_dim),
        )
        # to device
        self.to(device)

    def forward(self, obs: Tensor):
        return self.net(obs)
