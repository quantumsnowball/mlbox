import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch as T
import torch.optim as optim
from gymnasium.spaces import Box

from mlbox.agent.a2c import A2CContinuousAgent
from mlbox.agent.a2c.nn import ActorCriticContinuous

ENV = 'Pendulum-v1'

Obs = npt.NDArray[np.float32]
Action = npt.NDArray[np.float32]


class MyAgent(A2CContinuousAgent[Obs, Action]):
    ''' still not solved '''
    device = T.device('cpu')
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
        self.actor_critic_net = ActorCriticContinuous(in_dim, out_dim,
                                                      mu_clip=True,
                                                      mu_scale=2.0,
                                                      device=self.device)
        self.optimizer = optim.Adam(self.actor_critic_net.parameters(),
                                    lr=1e-2)


agent = MyAgent()
agent.prompt('model.pth')
agent.play(agent.max_step, env=agent.render_env)
