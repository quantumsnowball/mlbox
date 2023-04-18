from typing import Generic

from gymnasium import Env

from mlbox.types import T_Action, T_Obs
from mlbox.utils.wrapper import assured


class BasicAgentProps(Generic[T_Obs, T_Action]):
    #
    # env
    #

    @property
    @assured
    def env(self) -> Env[T_Obs, T_Action]:
        ''' a gym.Env compatible object '''
        return self._env

    @env.setter
    def env(self, env: Env[T_Obs, T_Action]) -> None:
        self._env = env

    @property
    @assured
    def render_env(self) -> Env[T_Obs, T_Action]:
        ''' a gym.Env compatible object for human render mode'''
        return self._render_env

    @render_env.setter
    def render_env(self, render_env: Env[T_Obs, T_Action]) -> None:
        self._render_env = render_env
