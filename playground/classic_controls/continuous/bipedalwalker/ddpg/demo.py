import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch as T
import torch.optim as optim
from gymnasium.spaces import Box

from mlbox.agent.ddpg import DDPGAgent
from mlbox.agent.ddpg.nn import DDPGActorNet, DDPGCriticNet

ENV = 'BipedalWalker-v3'

Obs = npt.NDArray[np.float32]
Action = npt.NDArray[np.float32]


class MyAgent(DDPGAgent[Obs, Action]):
    device = T.device('cuda')
    max_step = 500
    n_eps = 5000
    n_epoch = 10
    replay_size = 1000*max_step
    batch_size = 128
    update_target_every = 10
    print_hash_every = 5
    rolling_reward_ma = 20
    report_progress_every = 50
    render_every = 500

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        self.render_env = gym.make(ENV, render_mode='human')
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Box)
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        high = self.env.action_space.high
        low = self.env.action_space.low
        self.min_noise = 0.2
        self.max_noise = high * 5
        self.actor_net = DDPGActorNet(obs_dim, action_dim,
                                      min_action=low,
                                      max_action=high).to(self.device)
        self.actor_net_target = DDPGActorNet(obs_dim, action_dim,
                                             min_action=low,
                                             max_action=high).to(self.device)
        self.critic_net = DDPGCriticNet(obs_dim, action_dim).to(self.device)
        self.critic_net_target = DDPGCriticNet(obs_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=1e-3)


agent = MyAgent()
agent.prompt('model.pth')
agent.play(agent.max_step, env=agent.render_env)
