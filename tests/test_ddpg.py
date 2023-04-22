import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
import torch as T
import torch.optim as optim
from gymnasium.spaces import Box

from mlbox.agent.ddpg import DDPGAgent
from mlbox.agent.ddpg.nn import DDPGActorNet, DDPGCriticNet

ENV = 'Pendulum-v1'

Obs = npt.NDArray[np.float32]
Action = npt.NDArray[np.float32]


@pytest.mark.parametrize('run_on', ('cpu', 'cuda'))
def test_ddpg(run_on):
    class MyAgent(DDPGAgent[Obs, Action]):
        device = T.device(run_on)
        max_step = 500
        n_eps = 5
        n_epoch = 5
        replay_size = 1000*max_step
        batch_size = 256
        update_target_every = 5
        print_hash_every = 5
        rolling_reward_ma = 20
        report_progress_every = 50
        validation = True
        render_every = 1000
        min_noise = 0.5
        max_noise = 3.0

        def __init__(self) -> None:
            super().__init__()
            self.env = self.vald_env = gym.make(ENV)
            self.render_env = gym.make(ENV, render_mode='human')
            self.actor_net = DDPGActorNet(self.obs_dim, self.action_dim,
                                          min_action=-2,
                                          max_action=+2).to(self.device)
            self.actor_net_target = DDPGActorNet(self.obs_dim, self.action_dim,
                                                 min_action=-2,
                                                 max_action=+2).to(self.device)
            self.critic_net = DDPGCriticNet(self.obs_dim, self.action_dim).to(self.device)
            self.critic_net_target = DDPGCriticNet(self.obs_dim, self.action_dim).to(self.device)
            self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=5e-3)
            self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=5e-3)

    MyAgent().train()
