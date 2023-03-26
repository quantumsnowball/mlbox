from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread
from typing import Any, Generic, Self, SupportsFloat, TypeVar

from gymnasium import Env
from trbox.event.market import OhlcvWindow
from trbox.strategy import Context, Hook
from trbox.trader import Trader

T_Obs = TypeVar('T_Obs')
T_Action = TypeVar('T_Action')
T_Reward = TypeVar('T_Reward')


class TrEnv(Env[T_Obs, T_Action], Generic[T_Obs, T_Action, T_Reward], ABC):
    interval = 1

    def __new__(cls) -> Self:
        try:
            # ensure attrs are implemented in subclass instance
            cls.observation_space
            cls.action_space
            return super().__new__(cls)
        except AttributeError as e:
            raise NotImplementedError(e.name) from None

    def __init__(self) -> None:
        super().__init__()
        self.obs_q = Queue[T_Obs](maxsize=1)
        self.action_q = Queue[T_Action](maxsize=1)
        self.reward_q = Queue[T_Reward](maxsize=1)

    @abstractmethod
    def make(self) -> Trader:
        ...

    @abstractmethod
    def observe(self, my: Context[OhlcvWindow]) -> T_Obs:
        ...

    @abstractmethod
    def act(self, my: Context[OhlcvWindow], action: T_Action) -> None:
        ...

    @abstractmethod
    def grant(self, my: Context[OhlcvWindow]) -> T_Reward:
        ...

    def every(self, _: Context[OhlcvWindow]) -> None:
        pass

    def beginning(self, _: Context[OhlcvWindow]) -> None:
        pass

    @property
    def do(self) -> Hook[OhlcvWindow]:
        def do(my: Context[OhlcvWindow]) -> None:
            self.every(my)
            if my.count.beginning:
                self.beginning(my)
            elif my.count.every(self.interval):
                # observe
                obs = self.observe(my)
                self.obs_q.put(obs)
                # take action
                action = self.action_q.get()
                self.act(my, action)
                # collect experience
                reward = self.grant(my)
                self.reward_q.put(reward)
        return do

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
        trader = self.make()
        t = Thread(target=trader.run, daemon=True)
        t.start()

        obs = self.obs_q.get()
        info = {}
        return obs, info

    def step(self,
             action: T_Action) -> tuple[T_Obs,
                                        T_Reward,
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
        self.action_q.put(action)

        reward = self.reward_q.get()
        obs = self.obs_q.get()
        return obs, reward, False, False, {}
