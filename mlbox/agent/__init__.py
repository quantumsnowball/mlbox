from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Literal, Self

from gymnasium import Env

from mlbox.types import T_Action, T_Obs


class Agent(ABC, Generic[T_Obs, T_Action]):
    device: Literal['cuda', 'cpu', ]

    def __new__(cls) -> Self:
        try:
            # ensure attrs are implemented in subclass instance
            cls.device
            return super().__new__(cls)
        except AttributeError as e:
            raise NotImplementedError(e.name) from None

    def __init__(self) -> None:
        super().__init__()
        self.progress = 0.0
    #
    # env
    #

    @property
    def env(self) -> Env[T_Obs, T_Action]:
        ''' a gym.Env compatible object '''
        try:
            return self._env
        except AttributeError:
            raise NotImplementedError('env') from None

    @env.setter
    def env(self, env: Env[T_Obs, T_Action]) -> None:
        self._env = env

    #
    # acting
    #

    @abstractmethod
    def explore(self) -> T_Action:
        ''' take a random action '''
        ...

    @abstractmethod
    def exploit(self, obs: T_Obs) -> T_Action:
        ''' take an action decided by the policy '''
        ...

    @abstractmethod
    def decide(self,
               obs: T_Obs,
               *,
               epsilon: float) -> T_Action:
        ''' explore or exploit an action base on epsilon greedy algorithm '''
        ...

    #
    # training
    #

    @abstractmethod
    def learn(self,
              n_epoch: int,
              batch_size: int,
              gamma: float) -> None:
        ''' learn from replay experience '''
        ...

    @abstractmethod
    def train(self,
              n_eps: int) -> None:
        ''' train an agent to learn through all necessary steps '''
        ...

    #
    # I/O
    #

    @abstractmethod
    def load(self,
             path: Path | str) -> None:
        ...

    @abstractmethod
    def save(self,
             path: Path | str) -> None:
        ...

    @abstractmethod
    def prompt(self,
               name: str) -> None:
        ...
