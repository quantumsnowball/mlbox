import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.spaces import Box, Discrete
from torch import Tensor

from mlbox.agent.a2c import A2CDiscreteAgent
from mlbox.agent.a2c.nn import ActorCriticContinuous, ActorCriticDiscrete


@pytest.mark.parametrize('run_on', ('cpu', 'cuda'))
def test_a2c_discrete(run_on):
    env = 'CartPole-v1'

    Obs = npt.NDArray[np.float32]
    Action = np.int64

    class MyAgent(A2CDiscreteAgent[Obs, Action]):
        device = run_on
        max_step = 500
        n_eps = 5
        print_hash_every = 10
        rolling_reward_ma = 5
        report_progress_every = 100

        def __init__(self) -> None:
            super().__init__()
            self.env = gym.make(env)
            assert isinstance(self.env.observation_space, Box)
            assert isinstance(self.env.action_space, Discrete)
            in_dim = self.env.observation_space.shape[0]
            out_dim = self.env.action_space.n.item()
            self.actor_critic_net = ActorCriticDiscrete(in_dim, out_dim).to(self.device)
            self.optimizer = optim.Adam(self.actor_critic_net.parameters(),
                                        lr=1e-2)

    MyAgent().train()


@pytest.mark.parametrize('run_on', ('cpu', 'cuda'))
def test_a2c_continuous(run_on):
    env = 'Pendulum-v1'

    Obs = npt.NDArray[np.float32]
    Action = npt.NDArray[np.float32]

    class MyAgent(A2CDiscreteAgent[Obs, Action]):
        device = run_on
        max_step = 500
        n_eps = 5
        print_hash_every = 10
        rolling_reward_ma = 5
        report_progress_every = 100

        def __init__(self) -> None:
            super().__init__()
            self.env = gym.make(env)
            assert isinstance(self.env.observation_space, Box)
            assert isinstance(self.env.action_space, Box)
            in_dim = self.env.observation_space.shape[0]
            out_dim = self.env.action_space.shape[0]
            self.actor_critic_net = ActorCriticContinuous(in_dim, out_dim).to(self.device)
            self.optimizer = optim.Adam(self.actor_critic_net.parameters(),
                                        lr=1e-2)

    MyAgent().train()
