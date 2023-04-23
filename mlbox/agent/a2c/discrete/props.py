from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from mlbox.interface.agent import Agent
from mlbox.types import T_Action, T_Obs
from mlbox.utils.wrapper import assured


class Props(Agent[T_Obs, T_Action]):
    #
    # env
    #

    @property
    @assured
    def env(self) -> Env[T_Obs, T_Action]:
        return self._env

    @env.setter
    def env(self, env: Env[T_Obs, T_Action]) -> None:
        self._env = env
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Discrete)
        self.observation_space: Box = self.env.observation_space
        self.action_space: Discrete = self.env.action_space
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.n.item()
