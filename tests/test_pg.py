import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
import torch as T
from gymnasium.spaces import Box, Discrete
from torch.nn import ReLU, Tanh
from torch.optim import Adam

from mlbox.agent.pg import PGAgent
from mlbox.agent.pg.nn import BaselineNet, PolicyNet

ENV = 'CartPole-v1'

Obs = npt.NDArray[np.float32]
Action = np.int64


@pytest.mark.parametrize('run_on', ('cpu', 'cuda'))
def test_pg(run_on):
    class MyAgent(PGAgent[Obs, Action]):
        device = T.device(run_on)
        max_step = 500
        n_eps = 5
        batch_size = 30
        print_hash_every = 10
        report_progress_every = 5
        # variant
        reward_to_go = False
        baseline = True

        def __init__(self) -> None:
            super().__init__()
            self.env = gym.make(ENV)
            self.policy_net = PolicyNet(self.obs_dim, self.action_dim,
                                        hidden_dim=32,
                                        Activation=Tanh).to(self.device)
            self.baseline_net = BaselineNet(self.obs_dim, 1,
                                            hidden_dim=32,
                                            Activation=ReLU).to(self.device)
            self.policy_optimizer = Adam(self.policy_net.parameters(), lr=1e-2)
            self.baseline_optimizer = Adam(
                self.policy_net.parameters(), lr=1e-3)

    MyAgent().train()
