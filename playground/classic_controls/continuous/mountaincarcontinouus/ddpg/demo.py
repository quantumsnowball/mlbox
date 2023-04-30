import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch as T
import torch.optim as optim
from gymnasium.spaces import Box

from mlbox.agent.ddpg import DDPGAgent
from mlbox.agent.ddpg.nn import DDPGActorNet, DDPGCriticNet

ENV = 'MountainCarContinuous-v0'

Obs = npt.NDArray[np.float32]
Action = npt.NDArray[np.float32]


class MyAgent(DDPGAgent[Obs, Action]):
    device = T.device('cpu')
    max_step = 500
    n_eps = 5000
    n_epoch = 5
    replay_size = 1000*max_step
    batch_size = 256
    update_target_every = 5
    print_hash_every = 5
    rolling_reward_ma = 20
    report_progress_every = 50
    render_every = 1000
    min_noise = 1.5
    max_noise = 10.0

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
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


agent = MyAgent()
agent.prompt()
agent.play(agent.max_step, env=agent.render_env)
