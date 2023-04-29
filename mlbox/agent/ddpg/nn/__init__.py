from itertools import chain

import torch as T
import torch.nn.functional as F
from numpy import float32
from numpy.typing import NDArray
from torch import Tensor, tensor
from torch.nn import (BatchNorm1d, Dropout, Identity, Linear, Module,
                      Parameter, ReLU, Sequential)


class DDPGActorNet(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 hidden_dim: int = 256,
                 hidden_n: int = 1,
                 Activation: type[Module] = ReLU,
                 batch_norm: bool = True,
                 dropout: float | None = 0.0,
                 min_action: float | NDArray[float32] = -1,
                 max_action: float | NDArray[float32] = +1):
        super().__init__()
        # net
        self.net = Sequential(
            # input
            Linear(in_dim, hidden_dim),
            BatchNorm1d(hidden_dim) if batch_norm else Identity(),
            Activation(),
            Dropout(dropout) if dropout is not None else Identity(),
            # hidden
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation(),
                Dropout(dropout) if dropout is not None else Identity(),
            ) for _ in range(hidden_n))),
            # output
            Linear(hidden_dim, out_dim),
        )
        # const
        self.min_action = Parameter(tensor(min_action), requires_grad=False)
        self.max_action = Parameter(tensor(max_action), requires_grad=False)

    def forward(self, obs: Tensor) -> Tensor:
        x: Tensor = self.net(obs)
        x = F.sigmoid(x) * (self.max_action - self.min_action) + self.min_action
        return x


class DDPGCriticNet(Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 *,
                 hidden_dim: int = 256,
                 hidden_n: int = 0,
                 Activation: type[Module] = ReLU,
                 batch_norm: bool = True,
                 dropout: float | None = 0.0,):
        super().__init__()
        # obs
        self.obs_net = Sequential(
            Linear(obs_dim, hidden_dim),
            BatchNorm1d(hidden_dim) if batch_norm else Identity(),
            Activation(),
            Dropout(dropout) if dropout is not None else Identity(),
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation(),
                Dropout(dropout) if dropout is not None else Identity(),
            ) for _ in range(hidden_n)))
        )
        # action
        self.action_net = Sequential(
            Linear(action_dim, hidden_dim),
            BatchNorm1d(hidden_dim) if batch_norm else Identity(),
            Activation(),
            Dropout(dropout) if dropout is not None else Identity(),
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation(),
                Dropout(dropout) if dropout is not None else Identity(),
            ) for _ in range(hidden_n)))
        )
        # common
        self.common_net = Sequential(
            Linear(hidden_dim*2, hidden_dim),
            BatchNorm1d(hidden_dim) if batch_norm else Identity(),
            Activation(),
            Dropout(dropout) if dropout is not None else Identity(),
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation(),
                Dropout(dropout) if dropout is not None else Identity(),
            ) for _ in range(hidden_n))),
            Linear(hidden_dim, 1),
        )

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        obs_net_out = self.obs_net(obs)
        action_net_out = self.action_net(action)
        common_in = T.relu(T.cat([obs_net_out, action_net_out], dim=-1))
        q: Tensor = self.common_net(common_in)
        return q
