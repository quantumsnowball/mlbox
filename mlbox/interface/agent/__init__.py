from abc import abstractmethod
from pathlib import Path

from gymnasium import Env

from mlbox.interface.agent.props import AgentProps
from mlbox.types import T_Action, T_Obs


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
