from itertools import chain

import torch as T
import torch.nn.functional as F
from numpy import float32
from numpy.typing import NDArray
from torch import Tensor, tensor
from torch.nn import (LSTM, BatchNorm1d, Conv1d, Identity, Linear, Module,
                      Parameter, ReLU, Sequential)


class ConvLSTM_DDPGActorNet(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 hidden_dim: int = 256,
                 hidden_n: int = 1,
                 Activation: type[Module] = ReLU,
                 batch_norm: bool = True,
                 min_action: float | NDArray[float32] = -1,
                 max_action: float | NDArray[float32] = +1,
                 conv1d_in_channels: int = 1,
                 conv1d_out_channels: int = 16,
                 conv1d_kernel_size: int = 10,
                 conv1d_stride: int = 3,
                 conv1d_padding: int = 2,
                 lstm_hidden_dim: int = 64,
                 lstm_layers_n: int = 2):
        super().__init__()
        # const
        self.min_action = Parameter(tensor(min_action), requires_grad=False)
        self.max_action = Parameter(tensor(max_action), requires_grad=False)
        # conv1d
        self.conv1d = Conv1d(in_channels=conv1d_in_channels,
                             out_channels=conv1d_out_channels,
                             kernel_size=conv1d_kernel_size,
                             stride=conv1d_stride,
                             padding=conv1d_padding,
                             )
        self.conv1d_feat = (in_dim - self.conv1d.kernel_size[0] +
                            2 * int(self.conv1d.padding[0])) / self.conv1d.stride[0] + 1
        # lstm
        self.lstm = LSTM(input_size=conv1d_out_channels,
                         hidden_size=lstm_hidden_dim,
                         num_layers=lstm_layers_n,
                         batch_first=True)
        self.lstm_post = Sequential(
            BatchNorm1d(lstm_hidden_dim) if batch_norm else Identity(),
            Activation(),
        )
        self.lstm_feat = int(self.conv1d_feat*lstm_hidden_dim)
        # fc
        self.fc = Sequential(
            Linear(self.lstm_feat, hidden_dim),
            BatchNorm1d(hidden_dim) if batch_norm else Identity(),
            Activation(),
            # hidden
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation(),
            ) for _ in range(hidden_n))),
            # output
            Linear(hidden_dim, out_dim),
        )

    def forward(self, obs: Tensor) -> Tensor:
        x = obs.transpose(-2, -1)
        x = self.conv1d(x)
        x = x.transpose(-2, -1)
        x, _ = self.lstm(x)
        x = self.lstm_post(x.transpose(-2, -1))
        x = x.flatten(-2)
        x = self.fc(x)
        x = F.sigmoid(x) * (self.max_action - self.min_action) + self.min_action
        return x


class ConvLSTM_DDPGCriticNet(Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 *,
                 hidden_dim: int = 256,
                 hidden_n: int = 0,
                 Activation: type[Module] = ReLU,
                 batch_norm: bool = True,
                 conv1d_in_channels: int = 1,
                 conv1d_out_channels: int = 16,
                 conv1d_kernel_size: int = 10,
                 conv1d_stride: int = 3,
                 conv1d_padding: int = 2,
                 lstm_hidden_dim: int = 64,
                 lstm_layers_n: int = 2):
        super().__init__()
        # obs
        self.conv1d = Conv1d(in_channels=conv1d_in_channels,
                             out_channels=conv1d_out_channels,
                             kernel_size=conv1d_kernel_size,
                             stride=conv1d_stride,
                             padding=conv1d_padding,
                             )
        self.conv1d_feat = (obs_dim - self.conv1d.kernel_size[0] +
                            2 * int(self.conv1d.padding[0])) / self.conv1d.stride[0] + 1
        self.lstm = LSTM(input_size=conv1d_out_channels,
                         hidden_size=lstm_hidden_dim,
                         num_layers=lstm_layers_n,
                         batch_first=True)
        self.lstm_feat = int(self.conv1d_feat*lstm_hidden_dim)
        self.obs_net = Sequential(
            Linear(self.lstm_feat, hidden_dim),
            BatchNorm1d(hidden_dim) if batch_norm else Identity(),
            Activation(),
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation()
            ) for _ in range(hidden_n)))
        )
        # action
        self.action_net = Sequential(
            Linear(action_dim, hidden_dim),
            BatchNorm1d(hidden_dim) if batch_norm else Identity(),
            Activation(),
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation()
            ) for _ in range(hidden_n)))
        )
        # common
        self.common_net = Sequential(
            Linear(hidden_dim*2, hidden_dim),
            BatchNorm1d(hidden_dim) if batch_norm else Identity(),
            Activation(),
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation()
            ) for _ in range(hidden_n))),
            Linear(hidden_dim, 1),
        )

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        conv1d_out = self.conv1d(obs.transpose(-2, -1)).transpose(-2, -1)
        lstm_out, _ = self.lstm(conv1d_out)
        obs_net_out = self.obs_net(lstm_out.flatten(-2))
        action_net_out = self.action_net(action)
        common_in = T.relu(T.cat([obs_net_out, action_net_out], dim=-1))
        q: Tensor = self.common_net(common_in)
        return q
