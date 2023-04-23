from abc import ABC, abstractmethod

from gymnasium import Env
from trbox.trader import Trader

from mlbox.types import T_Action, T_Obs


class GymEnv(ABC, Env[T_Obs, T_Action]):
    @abstractmethod
    def make(self) -> Trader:
        ...
