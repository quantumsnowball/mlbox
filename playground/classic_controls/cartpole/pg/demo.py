from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.nn import Module
from torch.optim import Adam

from mlbox.agent.pg import PGAgent
from mlbox.neural import FullyConnected

ENV = 'CartPole-v1'

Obs = npt.NDArray[np.float32]
Action = np.int64


class PolicyNet(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._net = FullyConnected(*args, **kwargs)

    def forward(self, X):
        return Categorical(logits=self._net(X))


class MyAgent(PGAgent[Obs, Action]):
    device = 'cuda'
    max_step = 500
    n_eps = 1000
    batch_size = 6000

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Discrete)
        in_dim = self.env.observation_space.shape[0]
        out_dim = self.env.action_space.n.item()
        self.policy = PolicyNet(in_dim, out_dim, hidden_dim=32)).to(self.device)
        self.optimizer=Adam(self.policy.parameters(), lr = 1e-3)
