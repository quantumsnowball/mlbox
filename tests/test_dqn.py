import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
import torch as T
from torch.nn import MSELoss
from torch.optim import Adam

from mlbox.agent.dqn import DQNAgent
from mlbox.agent.dqn.nn import DQNNet

ENV = 'CartPole-v1'

Obs = npt.NDArray[np.float32]
Action = npt.NDArray[np.int64]


@pytest.mark.parametrize('run_on', ('cpu', 'cuda'))
def test_dqn(run_on):
    class MyAgent(DQNAgent[Obs, Action]):
        device = T.device(run_on)
        replay_size = 1000
        batch_size = 64
        max_step = 200
        update_target_every = 5
        report_progress_every = 20
        validation = True
        rolling_reward_ma = 5
        n_eps = 5
        n_epoch = 10
        gamma = 0.99

        def __init__(self) -> None:
            super().__init__()
            self.env = self.vald_env = gym.make(ENV)
            self.policy = DQNNet(self.obs_dim, self.action_dim,
                                 hidden_dim=32).to(self.device)
            self.target = DQNNet(self.obs_dim, self.action_dim,
                                 hidden_dim=32).to(self.device)
            self.update_target()
            self.optimizer = Adam(self.policy.parameters(),
                                  lr=1e-3)
            self.loss_function = MSELoss()

    MyAgent().train()
