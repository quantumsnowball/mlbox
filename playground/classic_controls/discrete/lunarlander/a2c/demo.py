import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch as T
import torch.optim as optim

from mlbox.agent.a2c.discrete import A2CDiscreteAgent
from mlbox.agent.a2c.nn import ActorCriticDiscrete

ENV = 'LunarLander-v2'

Obs = npt.NDArray[np.float32]
Action = npt.NDArray[np.int64]


class MyAgent(A2CDiscreteAgent[Obs, Action]):
    device = T.device('cpu')
    max_step = 1000
    n_eps = 5000
    print_hash_every = 10
    rolling_reward_ma = 5
    report_progress_every = 100
    render_every = 1000

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        self.render_env = gym.make(ENV, render_mode='human')
        self.actor_critic_net = ActorCriticDiscrete(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic_net.parameters(),
                                    lr=1e-2)


agent = MyAgent()
agent.prompt('model.pth')
agent.play(agent.max_step, env=agent.render_env)
