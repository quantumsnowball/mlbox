import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch.optim as optim
from gymnasium.spaces import Box, Discrete

from mlbox.agent.a2c import A2CDiscreteAgent
from mlbox.agent.a2c.nn import ActorCriticDiscrete

ENV = 'CartPole-v1'

Obs = npt.NDArray[np.float32]
Action = np.int64


class MyAgent(A2CDiscreteAgent[Obs, Action]):
    device = 'cpu'
    max_step = 1000
    n_eps = 2000
    print_hash_every = 10
    rolling_reward_ma = 5
    report_progress_every = 100
    render_every = 500

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        self.render_env = gym.make(ENV, render_mode='human')
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Discrete)
        in_dim = self.env.observation_space.shape[0]
        out_dim = self.env.action_space.n.item()
        self.actor_critic_net = ActorCriticDiscrete(in_dim, out_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic_net.parameters(),
                                    lr=1e-2)


agent = MyAgent()
agent.prompt('model.pth')
agent.play(agent.max_step, env=agent.render_env)
