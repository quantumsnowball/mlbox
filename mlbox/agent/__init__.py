from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic

from gymnasium import Env

from mlbox.types import T_Action, T_Obs


class Agent(ABC, Generic[T_Obs, T_Action]):
    '''
    Define the interface of an Agent
    '''

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

    @abstractmethod
    def play(self,
             max_step: int,
             *,
             env: Env[T_Obs, T_Action] | None = None) -> float:
        ''' agent to play through the env using current policy '''
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
