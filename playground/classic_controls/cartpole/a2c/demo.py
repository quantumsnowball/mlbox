import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.spaces import Box, Discrete
from torch import Tensor

from mlbox.agent.a2c import A2CAgent

ENV = 'CartPole-v1'

Obs = npt.NDArray[np.float32]
Action = np.int64


class ActorCriticNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, action_dim)
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs: Tensor):
        # action probs
        actor_x = F.relu(self.actor_fc1(obs))
        action_probs = F.softmax(self.actor_fc2(actor_x), dim=-1)
        # state value
        critic_x = F.relu(self.critic_fc1(obs))
        state_value = self.critic_fc2(critic_x)
        # return
        return action_probs, state_value


class MyAgent(A2CAgent[Obs, Action]):
    device = 'cpu'
    max_step = 500
    n_eps = 1500
    print_hash_every = 10
    rolling_reward_ma = 5
    report_progress_every = 100

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Discrete)
        in_dim = self.env.observation_space.shape[0]
        out_dim = self.env.action_space.n.item()
        self.actor_critic_net = ActorCriticNet(in_dim, out_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic_net.parameters(),
                                    lr=1e-2)


agent = MyAgent()
agent.prompt('model.pth')
agent.play(500, env=gym.make(ENV, render_mode='human'))
