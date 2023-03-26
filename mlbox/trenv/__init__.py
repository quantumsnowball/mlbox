from abc import ABC, abstractmethod
from typing import Any, Self, SupportsFloat, TypeVar

from gymnasium import Env, Space

T_Obs = TypeVar('T_Obs')
T_Action = TypeVar('T_Action')


class TrEnv(Env[T_Obs, T_Action], ABC):
    def __new__(cls) -> Self:
        try:
            # ensure attrs are implemented in subclass instance
            cls.observation_space
            cls.action_space
            return super().__new__(cls)
        except AttributeError as e:
            raise NotImplementedError(e.name) from None

    def __init__(self,
                 obs_space: Space[T_Obs],
                 action_space: Space[T_Action]) -> None:
        super().__init__()
        TrEnv.observation_space = obs_space
        TrEnv.action_space = action_space

    @abstractmethod
    def make(self):
        ...

    @abstractmethod
    def observe(self):
        ...

    @abstractmethod
    def act(self):
        ...

    @abstractmethod
    def grant(self):
        ...
    #
    # gym.Env
    #

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[T_Obs,
                                                              dict[str, Any]]:
        '''
        1. create a Trader, may be using make()
        2. start the Trader and run the first iteration until a heartbeat signal
        3. intercept observe() and get the first observation
        4. set the signal by step() and continue the iteration
        reset() needs to integrated with trbox strategy heartbeat events sync
        '''
        return tuple()

    def step(self,
             action: T_Action) -> tuple[T_Obs,
                                        SupportsFloat,
                                        bool,
                                        bool,
                                        dict[str, Any]]:
        '''
        1. accept an action and set the heartbeat signal
        2. wait for the next market data call incoming
        3. observe the next observation
        4. calculate the reward
        5. set the signal and return the result
        step() needs to integrated with trbox strategy heartbeat events sync
        '''
        return tuple()


print('TrEnv hi!')
