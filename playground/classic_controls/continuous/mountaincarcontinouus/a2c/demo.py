import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch as T
import torch.optim as optim

from mlbox.agent.a2c.continuous import A2CContinuousAgent
from mlbox.agent.a2c.nn import ActorCriticContinuous

ENV = 'MountainCarContinuous-v0'

Obs = npt.NDArray[np.float32]
Action = npt.NDArray[np.float32]


class MyAgent(A2CContinuousAgent[Obs, Action]):
    device = T.device('cpu')
    max_step = 200
    n_eps = 3000
    print_hash_every = 5
    rolling_reward_ma = 5
    report_progress_every = 50
    render_every = 250

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        self.render_env = gym.make(ENV, render_mode='human')
        self.actor_critic_net = ActorCriticContinuous(self.obs_dim, self.action_dim*2,
                                                      mu_clip=True,
                                                      mu_scale=2.0).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic_net.parameters(),
                                    lr=1e-2)


agent = MyAgent()
agent.prompt()
agent.play(agent.max_step, env=agent.render_env)
