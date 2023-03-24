from abc import ABC, abstractmethod
from typing import Generic, Literal, Self, TypeVar

from gymnasium import Space

T_State = TypeVar('T_State')
T_Action = TypeVar('T_Action')
T_Reward = TypeVar('T_Reward')


class Agent(ABC, Generic[T_State, T_Action, T_Reward]):
    action_space: Space[T_Action]
    observation_space: Space[T_State]
    device: Literal['cuda', 'cpu', ]

    def __new__(cls: type[Self]) -> Self:
        try:
            # ensure attrs are implemented in subclass instance
            cls.action_space
            cls.observation_space
            cls.device
            return super().__new__(cls)
        except AttributeError as e:
            raise NotImplementedError(e.name) from None

    #
    # acting
    #

    @abstractmethod
    def explore(self, state: T_State) -> T_Action:
        pass

    @abstractmethod
    def exploit(self, state: T_State) -> T_Action:
        pass

    @abstractmethod
    def decide(self,
               state: T_State,
               *,
               epilson: float = 0.5) -> T_Action:
        pass

    #
    # training
    #

    @abstractmethod
    def learn(self,
              epochs: int = 1000,
              batch_size: int = 512,
              gamma: float = 0.99) -> None:
        pass

    @abstractmethod
    def research(self) -> float:
        pass

    @abstractmethod
    def train(self,
              n_eps: int = 1000) -> None:
        pass
