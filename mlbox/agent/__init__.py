from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Literal, Self, TypeVar

from gymnasium import Env, Space

T_Obs = TypeVar('T_Obs')
T_Action = TypeVar('T_Action')
T_Reward = TypeVar('T_Reward')


class Agent(ABC, Generic[T_Obs, T_Action, T_Reward]):
    action_space: Space[T_Action]
    obs_space: Space[T_Obs]
    device: Literal['cuda', 'cpu', ]

    def __new__(cls,
                *,
                env: Env[T_Obs, T_Action]) -> Self:
        try:
            # ensure attrs are implemented in subclass instance
            cls.action_space
            cls.obs_space
            cls.device
            return super().__new__(cls)
        except AttributeError as e:
            raise NotImplementedError(e.name) from None

    def __init__(self,
                 *,
                 env: Env[T_Obs, T_Action]) -> None:
        super().__init__()
        self.env = env
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
