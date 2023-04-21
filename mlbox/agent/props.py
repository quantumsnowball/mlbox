from typing import Generic, Iterable

from gymnasium import Env
from torch.nn import Module
from torch.utils.tensorboard.writer import SummaryWriter

from mlbox.agent import AgentProps
from mlbox.types import T_Action, T_Obs
from mlbox.utils.wrapper import assured


class BasicAgentProps(AgentProps[T_Obs, T_Action]):
    #
    # env
    #

    @property
    @assured
    def render_env(self) -> Env[T_Obs, T_Action]:
        ''' a gym.Env compatible object for human render mode'''
        return self._render_env

    @render_env.setter
    def render_env(self, render_env: Env[T_Obs, T_Action]) -> None:
        self._render_env = render_env

    #
    # tensorboard
    #
    @property
    @assured
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, writer: SummaryWriter) -> None:
        self._writer = writer
