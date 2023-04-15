import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
from gymnasium.spaces import Box, Discrete
from torch.nn import Identity, ReLU, Tanh
from torch.optim import Adam

from mlbox.agent.pg import PGAgent
from mlbox.neural import FullyConnected

ENV = 'CartPole-v1'

Obs = npt.NDArray[np.float32]
Action = np.int64


@pytest.mark.parametrize('run_on', ('cpu', 'cuda'))
def test_pg(run_on):
    class MyAgent(PGAgent[Obs, Action]):
        device = run_on
        max_step = 500
        n_eps = 200
        batch_size = 3000
        print_hash_every = 10
        report_progress_every = 5
        # variant
        reward_to_go = False
        baseline = True

        def __init__(self) -> None:
            super().__init__()
            self.env = gym.make(ENV)
            assert isinstance(self.env.observation_space, Box)
            assert isinstance(self.env.action_space, Discrete)
            in_dim = self.env.observation_space.shape[0]
            out_dim = self.env.action_space.n.item()
            self.policy_net = FullyConnected(in_dim, out_dim,
                                             hidden_dim=32,
                                             Activation=Tanh,
                                             OutputActivation=Identity).to(self.device)
            self.baseline_net = FullyConnected(in_dim, 1,
                                               hidden_dim=32,
                                               Activation=ReLU).to(self.device)
            self.policy_optimizer = Adam(self.policy_net.parameters(), lr=1e-2)
            self.baseline_optimizer = Adam(
                self.policy_net.parameters(), lr=1e-3)

    MyAgent()
