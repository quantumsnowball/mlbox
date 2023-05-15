from abc import ABC, abstractmethod
from typing import Generic

from mlbox.types import T_Action, T_Obs


class Acting(ABC, Generic[T_Obs, T_Action]):
    #
    # Observing
    #

    @abstractmethod
    def encode_obs(self,
                   obs: T_Obs) -> T_Obs:
        ''' convert raw obs to encoded obs '''
        ...

    #
    # acting
    #

    @abstractmethod
    def decide(self,
               obs: T_Obs) -> T_Action:
        ''' given an observation choose an action '''
        ...
