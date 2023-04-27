from abc import ABC, abstractmethod
from typing import Generic

from gymnasium import Env
from torch.nn import Module

from mlbox.types import T_Action, T_Obs


class Network(ABC, Generic[T_Obs, T_Action]):
    #
    # neural nets
    #
    @property
    @abstractmethod
    def neural_nets(self) -> dict[str, Module]:
        ...

    @abstractmethod
    def use_train_mode(self) -> None:
        ...

    @abstractmethod
    def use_eval_mode(self) -> None:
        ...
