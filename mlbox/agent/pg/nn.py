from itertools import chain

from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Linear, Module, ReLU, Sequential, Tanh


class PolicyNet(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 hidden_dim: int = 64,
                 hidden_n: int = 1,
                 Activation: type[Module] = Tanh):
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
            # output layer
            Linear(hidden_dim, out_dim),
        )
        # dist
        self.dist = Categorical

    def forward(self, obs: Tensor) -> Categorical:
        logits = self.net(obs)
        policy = self.dist(logits=logits)
        return policy


class BaselineNet(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 hidden_dim: int = 64,
                 hidden_n: int = 1,
                 Activation: type[Module] = ReLU):
        super().__init__()
        self.net = Sequential(
            # input
            Linear(in_dim, hidden_dim),
            Activation(),
            # hidden
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                Activation(),
            ) for _ in range(hidden_n))),
            # output layer
            Linear(hidden_dim, out_dim),
        )

    def forward(self, obs: Tensor) -> Tensor:
        value: Tensor = self.net(obs)
        return value
