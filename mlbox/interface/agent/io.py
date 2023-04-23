from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic

from mlbox.types import T_Action, T_Obs


class IO(ABC, Generic[T_Obs, T_Action]):
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
