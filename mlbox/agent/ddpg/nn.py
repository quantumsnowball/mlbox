import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential


class DDPGActorNet(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 hidden_dim: int = 256,
                 hidden_n: int = 1,
                 Activation: type[Module] = ReLU,
                 min_action: float = -1,
                 max_action: float = +1):
        self.net = Sequential()
        self.net.append(Linear(in_dim, hidden_dim))
        self.net.append(Activation())
        for _ in range(hidden_n):
            self.net.append(Linear(hidden_dim, hidden_dim))
            self.net.append(Activation())
        self.net.append(Linear(hidden_dim, out_dim))
        self.max_action = max_action
        self.min_action = min_action

    def forward(self, obs: Tensor):
        x = self.net(obs)
        action = F.sigmoid(x) * (self.max_action - self.min_action) + self.min_action
        return action


class DDPGCriticNet(Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 *,
                 hidden_dim: int = 256,
                 hidden_n: int = 1,
                 Activation: type[Module] = ReLU):
        # obs
        self.obs_net = Sequential()
        self.obs_net.append(Linear(obs_dim, hidden_dim))
        self.obs_net.append(Activation())
        for _ in range(hidden_n):
            self.obs_net.append(Linear(hidden_dim, hidden_dim))
            self.obs_net.append(Activation())
        # obs
        self.action_net = Sequential()
        self.action_net.append(Linear(action_dim, hidden_dim))
        self.action_net.append(Activation())
        for _ in range(hidden_n):
            self.action_net.append(Linear(hidden_dim, hidden_dim))
            self.action_net.append(Activation())
        # common
        self.common_net = Sequential()
        self.common_net.append(Linear(hidden_dim*2, hidden_dim))
        for _ in range(hidden_n):
            self.common_net.append(Linear(hidden_dim, hidden_dim))
            self.common_net.append(Activation())
        self.common_net.append(Linear(hidden_dim, 1))

    def forward(self, obs: Tensor, action: Tensor):
        obs_net_out = self.obs_net(obs)
        action_net_out = self.action_net(action)
        common_in = torch.cat([obs_net_out, action_net_out], dim=1)
        q = self.common_net(common_in)
        return q
