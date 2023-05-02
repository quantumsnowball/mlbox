import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch as T
from torch.optim import Adam

from mlbox.agent.pg import PGAgent
from mlbox.agent.pg.nn import BaselineNet, PolicyNet

ENV = 'LunarLander-v2'

Obs = npt.NDArray[np.float32]
Action = npt.NDArray[np.int64]


class MyAgent(PGAgent[Obs, Action]):
    device = T.device('cpu')
    max_step = 1000
    n_eps = 1000
    batch_size = 3000
    print_hash_every = 10
    report_progress_every = 5
    render_every = 100
    # variant
    reward_to_go = True
    baseline = True

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        self.render_env = gym.make(ENV, render_mode='human')
        self.policy_net = PolicyNet(self.obs_dim, self.action_dim).to(self.device)
        self.baseline_net = BaselineNet(self.obs_dim, 1).to(self.device)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=1e-3)
        self.baseline_optimizer = Adam(self.baseline_net.parameters(), lr=1e-3)


agent = MyAgent()
agent.prompt()
agent.play(agent.max_step, env=agent.render_env)
