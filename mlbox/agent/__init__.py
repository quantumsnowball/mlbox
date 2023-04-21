from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic

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


class Agent(AgentProps[T_Obs, T_Action]):
    '''
    Define the interface of an Agent
    '''

    #
    # acting
    #

    @abstractmethod
    def decide(self,
               obs: T_Obs) -> T_Action:
        ''' given an observation choose an action '''
        ...

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


class EpsilonGreedyStrategy(ABC, Generic[T_Obs, T_Action]):
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
               epsilon: float = 1) -> T_Action:
        ''' explore or exploit an action base on epsilon greedy algorithm '''
        ...
