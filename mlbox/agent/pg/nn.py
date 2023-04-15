import torch as T
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Identity, Linear, Module, ReLU, Sequential, Tanh


class PolicyNet(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 device: T.device,
                 hidden_dim: int = 64,
                 hidden_n: int = 1,
                 Activation: type[Module] = Tanh):
        super().__init__()
        self.net = Sequential()
        # input layer
        self.net.append(Linear(in_dim, hidden_dim))
        self.net.append(Activation())
        # hidden layers
        for _ in range(hidden_n):
            self.net.append(Linear(hidden_dim, hidden_dim))
            self.net.append(Activation())
        # output layer
        self.net.append(Linear(hidden_dim, out_dim))
        # dist
        self.dist = Categorical
        # to device
        self.to(device)

    def forward(self, obs: Tensor):
        logits = self.net(obs)
        policy = self.dist(logits=logits)
        return policy


class BaselineNet(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 device: T.device,
                 hidden_dim: int = 64,
                 hidden_n: int = 1,
                 Activation: type[Module] = ReLU):
        super().__init__()
        self.net = Sequential()
        # input layer
        self.net.append(Linear(in_dim, hidden_dim))
        self.net.append(Activation())
        # hidden layers
        for _ in range(hidden_n):
            self.net.append(Linear(hidden_dim, hidden_dim))
            self.net.append(Activation())
        # output layer
        self.net.append(Linear(hidden_dim, out_dim))
        # to device
        self.to(device)

    def forward(self, obs: Tensor):
        value = self.net(obs)
        return value
