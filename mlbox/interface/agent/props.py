from abc import ABC, abstractmethod
from typing import Generic

from gymnasium import Env
from torch.utils.tensorboard.writer import SummaryWriter

from mlbox.types import T_Action, T_Obs


class AgentProps(ABC, Generic[T_Obs, T_Action]):
    #
    # env
    #

    @property
    @abstractmethod
    def env(self) -> Env[T_Obs, T_Action]:
        ''' a gym.Env compatible object '''
        ...

    @env.setter
    @abstractmethod
    def env(self,
            env: Env[T_Obs, T_Action]) -> None:
        ...

    @property
    @abstractmethod
    def render_env(self) -> Env[T_Obs, T_Action]:
        ''' a gym.Env compatible object for human render mode'''
        ...

    @render_env.setter
    @abstractmethod
    def render_env(self,
                   render_env: Env[T_Obs, T_Action]) -> None:
        ...

    #
    # tensorboard
    #
    @property
    @abstractmethod
    def writer(self) -> SummaryWriter:
        ...

    @writer.setter
    @abstractmethod
    def writer(self,
               writer: SummaryWriter) -> None:
        ...
