import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Box, Discrete
from torch import Tensor

from mlbox.agent.a2c import A2CAgent
from mlbox.agent.a2c.nn import ActorCriticNet

ENV = 'CartPole-v1'

Obs = npt.NDArray[np.float32]
Action = np.int64


class MyAgent(A2CAgent[Obs, Action]):
    device = 'cpu'
    max_step = 500
    n_eps = 2000
    print_hash_every = 10
    rolling_reward_ma = 5
    report_progress_every = 100

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Discrete)
        in_dim = self.env.observation_space.shape[0]
        out_dim = self.env.action_space.n.item()
        self.actor_critic_net = ActorCriticNet(in_dim, out_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic_net.parameters(),
                                    lr=1e-2)


agent = MyAgent()
agent.prompt('model.pth')
agent.play(500, env=gym.make(ENV, render_mode='human'))
