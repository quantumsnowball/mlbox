import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch.optim as optim
from gymnasium.spaces import Box

from mlbox.agent.ddpg import DDPGAgent
from mlbox.agent.ddpg.nn import DDPGActorNet, DDPGCriticNet

ENV = 'Pendulum-v1'

Obs = npt.NDArray[np.float32]
Action = npt.NDArray[np.float32]


class MyAgent(DDPGAgent[Obs, Action]):
    ''' still not solved '''
    device = 'cpu'
    max_step = 500
    n_eps = 5000
    print_hash_every = 5
    rolling_reward_ma = 5
    report_progress_every = 50
    render_every = 500

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        self.render_env = gym.make(ENV, render_mode='human')
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Box)
        in_dim = self.env.observation_space.shape[0]
        out_dim = self.env.action_space.shape[0]*2
        self.actor_net = DDPGActorNet(in_dim, out_dim).to(self.device)
        self.actor_net_target = DDPGActorNet(in_dim, out_dim).to(self.device)
        self.critic_net = DDPGCriticNet(in_dim, out_dim).to(self.device)
        self.critic_net_target = DDPGCriticNet(in_dim, out_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=1e-3)


agent = MyAgent()
agent.prompt('model.pth')
agent.play(agent.max_step, env=agent.render_env)
