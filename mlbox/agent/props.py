import sys
from pathlib import Path
from typing import Generic

from gymnasium import Env
from torch.utils.tensorboard.writer import SummaryWriter

from mlbox.types import T_Action, T_Obs
from mlbox.utils.wrapper import assured


class BasicAgentProps(Generic[T_Obs, T_Action]):
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

    @property
    @assured
    def vald_env(self) -> Env[T_Obs, T_Action]:
        ''' a gym.Env compatible object for validation '''
        return self._vald_env

    @vald_env.setter
    def vald_env(self, vald_env: Env[T_Obs, T_Action]) -> None:
        self._vald_env = vald_env

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
    #
    # I/O
    #

    @property
    def script_basedir(self) -> Path:
        path = Path().cwd() / Path(sys.argv[0])
        assert path.is_file()
        basedir = Path(path).parent.relative_to(Path.cwd())
        return basedir
