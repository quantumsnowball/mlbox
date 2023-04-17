from itertools import chain
from typing import Self

import torch as T
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, tensor
from torch.nn import Linear, Module, ReLU, Sequential
from typing_extensions import override


class DDPGActorNet(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 device: T.device,
                 hidden_dim: int = 256,
                 hidden_n: int = 1,
                 Activation: type[Module] = ReLU,
                 min_action: float | NDArray = -1,
                 max_action: float | NDArray = +1):
        super().__init__()
        self.device = device
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
        # const
        self.min_action = tensor(min_action, device=device)
        self.max_action = tensor(max_action, device=device)
        # to device
        self.to(device)

    def forward(self, obs: Tensor):
        x = self.net(obs)
        x = F.sigmoid(x) * (self.max_action - self.min_action) + self.min_action
        return x


class DDPGCriticNet(Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 *,
                 device: T.device,
                 hidden_dim: int = 256,
                 hidden_n: int = 0,
                 Activation: type[Module] = ReLU):
        super().__init__()
        # obs
        self.obs_net = Sequential(
            Linear(obs_dim, hidden_dim),
            Activation(),
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                Activation()
            ) for _ in range(hidden_n)))
        )
        # action
        self.action_net = Sequential(
            Linear(action_dim, hidden_dim),
            Activation(),
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                Activation()
            ) for _ in range(hidden_n)))
        )
        # common
        self.common_net = Sequential(
            Linear(hidden_dim*2, hidden_dim),
            Activation(),
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                Activation()
            ) for _ in range(hidden_n))),
            Linear(hidden_dim, 1),
        )
        # to device
        self.to(device)

    def forward(self, obs: Tensor, action: Tensor):
        obs_net_out = self.obs_net(obs)
        action_net_out = self.action_net(action)
        common_in = T.relu(T.cat([obs_net_out, action_net_out], dim=1))
        q = self.common_net(common_in)
        return q
