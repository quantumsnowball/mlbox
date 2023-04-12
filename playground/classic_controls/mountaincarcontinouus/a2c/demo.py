import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch.optim as optim
from gymnasium.spaces import Box

from mlbox.agent.a2c import A2CContinuousAgent
from mlbox.agent.a2c.nn import ActorCriticContinuous

ENV = 'MountainCarContinuous-v0'

Obs = npt.NDArray[np.float32]
Action = np.int64


class MyAgent(A2CContinuousAgent[Obs, Action]):
    device = 'cuda'
    max_step = 100
    n_eps = 1500
    print_hash_every = 5
    rolling_reward_ma = 5
    report_progress_every = 50

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Box)
        in_dim = self.env.observation_space.shape[0]
        out_dim = self.env.action_space.shape[0]*2
        self.actor_critic_net = ActorCriticContinuous(in_dim, out_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic_net.parameters(),
                                    lr=1e-3)


agent = MyAgent()
agent.prompt('model.pth')
agent.play(500, env=gym.make(ENV, render_mode='human'))
