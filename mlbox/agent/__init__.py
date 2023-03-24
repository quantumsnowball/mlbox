from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T_State = TypeVar('T_State')
T_Action = TypeVar('T_Action')


class Agent(ABC, Generic[T_State, T_Action]):
    def __init__(self) -> None:
        pass

    #
    # acting
    #

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
    def update_target(self) -> None:
        pass

    @abstractmethod
    def learn(self,
              epochs: int = 1000,
              batch_size: int = 512,
              gamma: float = 0.99) -> None:
        pass

    @abstractmethod
    def explore(self) -> float:
        pass

    @abstractmethod
    def train(self,
              n_eps: int = 1000) -> None:
        pass
