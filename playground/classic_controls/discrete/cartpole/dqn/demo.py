import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch as T
from torch.nn import MSELoss
from torch.optim import Adam

from mlbox.agent.dqn import DQNAgent
from mlbox.agent.dqn.nn import DQNNet

ENV = 'CartPole-v1'

Obs = npt.NDArray[np.float32]
Action = np.int64


class MyAgent(DQNAgent[Obs, Action]):
    device = T.device('cpu')
    replay_size = 5000
    batch_size = 64
    max_step = 1000
    n_eps = 5000
    n_epoch = 10
    gamma = 0.95
    rolling_reward_ma = 20
    update_target_every = 10
    print_hash_every = 5
    report_progress_every = 100
    render_every = 1000

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        self.render_env = gym.make(ENV, render_mode='human')
        self.policy = DQNNet(self.obs_dim, self.action_dim,
                             hidden_dim=64,
                             hidden_n=0).to(self.device)
        self.target = DQNNet(self.obs_dim, self.action_dim,
                             hidden_dim=64,
                             hidden_n=0).to(self.device)
        self.update_target()
        self.optimizer = Adam(self.policy.parameters(),
                              lr=2e-3)
        self.loss_function = MSELoss()


agent = MyAgent()
agent.prompt('model.pth',)
agent.play(agent.max_step, env=agent.render_env)
