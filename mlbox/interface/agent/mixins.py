from abc import ABC, abstractmethod
from typing import Generic

from mlbox.types import T_Action, T_Obs


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
