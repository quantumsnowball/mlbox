import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
import torch as T
import torch.optim as optim

from mlbox.agent.a2c.continuous import A2CContinuousAgent
from mlbox.agent.a2c.discrete import A2CDiscreteAgent
from mlbox.agent.a2c.nn import ActorCriticContinuous, ActorCriticDiscrete


@pytest.mark.parametrize('run_on', ('cpu', 'cuda'))
def test_a2c_discrete(run_on):
    env = 'CartPole-v1'

    Obs = npt.NDArray[np.float32]
    Action = npt.NDArray[np.int64]

    class MyAgent(A2CDiscreteAgent[Obs, Action]):
        device = T.device(run_on)
        max_step = 500
        n_eps = 5
        print_hash_every = 10
        rolling_reward_ma = 5
        report_progress_every = 100
        validation = True

        def __init__(self) -> None:
            super().__init__()
            self.env = self.vald_env = gym.make(env)
            self.actor_critic_net = ActorCriticDiscrete(self.obs_dim, self.action_dim).to(self.device)
            self.optimizer = optim.Adam(self.actor_critic_net.parameters(),
                                        lr=1e-2)

    MyAgent().train()


@pytest.mark.parametrize('run_on', ('cpu', 'cuda'))
def test_a2c_continuous(run_on):
    env = 'Pendulum-v1'

    Obs = npt.NDArray[np.float32]
    Action = npt.NDArray[np.float32]

    class MyAgent(A2CContinuousAgent[Obs, Action]):
        device = T.device(run_on)
        max_step = 500
        n_eps = 5
        print_hash_every = 10
        rolling_reward_ma = 5
        report_progress_every = 100

        def __init__(self) -> None:
            super().__init__()
            self.env = gym.make(env)
            self.actor_critic_net = ActorCriticContinuous(self.obs_dim, self.action_dim*2).to(self.device)
            self.optimizer = optim.Adam(self.actor_critic_net.parameters(),
                                        lr=1e-2)

    MyAgent().train()
