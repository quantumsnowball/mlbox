from abc import ABC, abstractmethod
from typing import Generic, SupportsFloat

from trbox.event.market import OhlcvWindow
from trbox.strategy.context import Context
from trbox.strategy.types import Hook

from mlbox.types import T_Action, T_Obs


class Routine(ABC, Generic[T_Obs, T_Action]):
    @abstractmethod
    def observe(self, my: Context[OhlcvWindow]) -> T_Obs:
        ...

    @abstractmethod
    def act(self, my: Context[OhlcvWindow], action: T_Action) -> None:
        ...

    @abstractmethod
    def grant(self, my: Context[OhlcvWindow]) -> SupportsFloat:
        ...

    @abstractmethod
    def every(self, _: Context[OhlcvWindow]) -> None:
        pass

    @abstractmethod
    def beginning(self, _: Context[OhlcvWindow]) -> None:
        pass

    @property
    @abstractmethod
    def do(self) -> Hook[OhlcvWindow]:
        ...
