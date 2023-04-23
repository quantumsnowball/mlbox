from abc import ABC, abstractmethod
from typing import Generic

from torch.utils.tensorboard.writer import SummaryWriter

from mlbox.types import T_Action, T_Obs


class Tensorboard(ABC, Generic[T_Obs, T_Action]):
    #
    # tensorboard
    #
    @property
    @abstractmethod
    def writer(self) -> SummaryWriter:
        ...

    @writer.setter
    @abstractmethod
    def writer(self,
               writer: SummaryWriter) -> None:
        ...
