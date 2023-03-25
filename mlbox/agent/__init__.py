from abc import ABC, abstractmethod
from typing import Generic, Literal, Self, TypeVar

from gymnasium import Space
from trbox.strategy import Hook
from trbox.trader import Trader

T_State = TypeVar('T_State')
T_Action = TypeVar('T_Action')
T_Reward = TypeVar('T_Reward')


class Agent(ABC, Generic[T_State, T_Action, T_Reward]):
    action_space: Space[T_Action]
    observation_space: Space[T_State]
    device: Literal['cuda', 'cpu', ]

    def __new__(cls) -> Self:
        try:
            # ensure attrs are implemented in subclass instance
            cls.action_space
            cls.observation_space
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

    @abstractmethod
    def make(self) -> Trader:
        ''' create a new env '''
        ...

    def reset(self) -> None:
        ''' calling self.make() to reset self.env to a new one '''
        self.env = self.make()

    @property
    @abstractmethod
    def explorer(self) -> Hook:
        ''' factory method to create the Hook for the Trader env '''
        ...

    @property
    def env(self) -> Trader:
        ''' Trader as env '''
        try:
            return self._env
        except AttributeError:
            raise NotImplementedError('env') from None

    @env.setter
    def env(self, env: Trader) -> None:
        self._env = env

    #
    # acting
    #

    @abstractmethod
    def explore(self) -> T_Action:
        ''' take a random action '''
        ...

    @abstractmethod
    def exploit(self, state: T_State) -> T_Action:
        ''' take an action decided by the policy '''
        ...

    @abstractmethod
    def decide(self,
               state: T_State,
               *,
               epilson: float) -> T_Action:
        ''' explore or exploit an action base on epsilon greedy algorithm '''
        ...

    #
    # training
    #

    @abstractmethod
    def learn(self,
              epochs: int,
              batch_size: int,
              gamma: float) -> None:
        ''' learn from replay experience '''
        ...

    @abstractmethod
    def train(self,
              n_eps: int) -> None:
        ''' train an agent to learn through all necessary steps '''
        ...
