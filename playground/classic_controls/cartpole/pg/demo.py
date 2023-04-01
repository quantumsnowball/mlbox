import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, Discrete
from torch.optim import Adam

from mlbox.agent.pg import PGAgent
from mlbox.neural import FullyConnected

ENV = 'CartPole-v1'

Obs = npt.NDArray[np.float32]
Action = np.int64


class MyAgent(PGAgent[Obs, Action]):
    device = 'cpu'
    max_step = 500
    n_eps = 1000
    batch_size = 6000

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(ENV)
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Discrete)
        in_dim = self.env.observation_space.shape[0]
        out_dim = self.env.action_space.n.item()
        self.policy_net = FullyConnected(
            in_dim, out_dim, hidden_dim=32).to(self.device)
        self.optimizer = Adam(self.policy_net.parameters(), lr=1e-3)


agent = MyAgent()
agent.prompt('model.pth', start_training=True)
