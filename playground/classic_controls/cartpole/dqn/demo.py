import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, Discrete
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from mlbox.agent.dqn import DQNAgent
from mlbox.neural import FullyConnected

Obs = npt.NDArray[np.float32]
Action = np.int64


class MyAgent(DQNAgent[Obs, Action]):
    device = 'cuda'
    replay_size = 10000
    batch_size = 512
    update_target_every = 20
    report_progress_every = 20
    rolling_reward_ma = 20
    n_eps = 3000
    n_epoch = 50
    gamma = 0.9

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make('CartPole-v1')
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Discrete)
        in_dim = self.env.observation_space.shape[0]
        out_dim = self.env.action_space.n.item()
        self.policy = FullyConnected(in_dim, out_dim).to(self.device)
        self.target = FullyConnected(in_dim, out_dim).to(self.device)
        self.update_target()
        self.optimizer = Adam(self.policy.parameters(),
                              lr=1e-3)
        self.loss_function = CrossEntropyLoss()


agent = MyAgent()
agent.prompt('model.pth', start_training=True)
