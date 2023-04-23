from abc import ABC, abstractmethod
from typing import Generic

from gymnasium import Env

from mlbox.types import T_Action, T_Obs


class Training(ABC, Generic[T_Obs, T_Action]):
    #
    # training
    #

    @abstractmethod
    def learn(self) -> None:
        ''' learn from replay experience '''
        ...

    @abstractmethod
    def train(self) -> None:
        ''' train an agent to learn through all necessary steps '''
        ...

    @abstractmethod
    def play(self,
             max_step: int,
             *,
             env: Env[T_Obs, T_Action] | None = None) -> float:
        ''' agent to play through the env using current policy '''
        ...
