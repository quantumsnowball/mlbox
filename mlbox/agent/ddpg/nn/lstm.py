from itertools import chain
from typing import Any

import torch as T
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, tensor
from torch.nn import LSTM, Linear, Module, Parameter, ReLU, Sequential


class LSTM_DDPGActorNet(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 hidden_dim: int = 256,
                 hidden_n: int = 1,
                 Activation: type[Module] = ReLU,
                 min_action: float | NDArray = -1,
                 max_action: float | NDArray = +1,
                 lstm_hidden_dim: int = 64,
                 lstm_layers_n: int = 2):
        super().__init__()
        # const
        self.lstm_L = in_dim
        self.lstm_feat = in_dim*lstm_hidden_dim
        self.min_action = Parameter(tensor(min_action), requires_grad=False)
        self.max_action = Parameter(tensor(max_action), requires_grad=False)
        # lstm
        self.lstm = LSTM(input_size=1,
                         hidden_size=lstm_hidden_dim,
                         num_layers=lstm_layers_n,
                         batch_first=True)
        # fc
        self.fc = Sequential(
            Linear(self.lstm_feat, hidden_dim),
            Activation(),
            # hidden
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                Activation(),
            ) for _ in range(hidden_n))),
            # output
            Linear(hidden_dim, out_dim),
        )

    def forward(self, obs: Tensor):
        x = obs.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = x.flatten(-2)
        x = self.fc(x)
        x = F.sigmoid(x) * (self.max_action - self.min_action) + self.min_action
        return x


class LSTM_DDPGCriticNet(Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 *,
                 hidden_dim: int = 256,
                 hidden_n: int = 0,
                 Activation: type[Module] = ReLU,
                 lstm_hidden_dim: int = 64,
                 lstm_layers_n: int = 2):
        super().__init__()
        # const
        self.lstm_L = obs_dim
        self.lstm_feat = obs_dim*lstm_hidden_dim
        # obs
        self.lstm = LSTM(input_size=1,
                         hidden_size=lstm_hidden_dim,
                         num_layers=lstm_layers_n,
                         batch_first=True)
        self.obs_net = Sequential(
            Linear(self.lstm_feat, hidden_dim),
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

    def forward(self, obs: Tensor, action: Tensor):
        lstm_out, _ = self.lstm(obs.unsqueeze(-1))
        obs_net_out = self.obs_net(lstm_out.flatten(-2))
        action_net_out = self.action_net(action)
        common_in = T.relu(T.cat([obs_net_out, action_net_out], dim=-1))
        q = self.common_net(common_in)
        return q
