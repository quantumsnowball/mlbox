import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch as T
from gymnasium.spaces import Box, Discrete
from torch.optim import Adam

from mlbox.agent.pg import PGAgent
from mlbox.agent.pg.nn import BaselineNet, PolicyNet

ENV = 'CartPole-v1'

Obs = npt.NDArray[np.float32]
Action = np.int64


class MyAgent(PGAgent[Obs, Action]):
    device = T.device('cpu')
    max_step = 1000
    n_eps = 200
    batch_size = 3000
    print_hash_every = 10
    report_progress_every = 5
    render_every = 50
    # variant
    reward_to_go = True
    baseline = True

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        self.render_env = gym.make(ENV, render_mode='human')
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Discrete)
        in_dim = self.env.observation_space.shape[0]
        out_dim = self.env.action_space.n.item()
        self.policy_net = PolicyNet(in_dim, out_dim, device=self.device)
        self.baseline_net = BaselineNet(in_dim, 1, device=self.device)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=1e-3)
        self.baseline_optimizer = Adam(self.baseline_net.parameters(), lr=1e-3)


agent = MyAgent()
agent.prompt('model.pth', )
agent.play(agent.max_step, env=agent.render_env)
