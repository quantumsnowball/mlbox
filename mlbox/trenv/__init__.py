from abc import ABC, abstractmethod
from queue import Queue
from threading import Event, Thread
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
        self.obs_q = Queue[T_Obs]()
        self.action_q = Queue[T_Action]()
        self.reward_q = Queue[T_Reward]()
        self._ready = Event()

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
                # collect experience if step
                if self._ready.is_set():
                    reward = self.grant(my)
                    self.reward_q.put(reward)
                # observe
                obs = self.observe(my)
                self.obs_q.put(obs)
                # take action
                action = self.action_q.get()
                self.act(my, action)
        return do

    #
    # gym.Env
    #

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[T_Obs,
                                                              dict[str, Any]]:
        # close old env if exists
        try:
            self._trader.stop()
        except AttributeError:
            pass
        # clear flag until first step
        self._ready.clear()
        # create env
        self._trader = self.make()
        t = Thread(target=self._trader.run, daemon=True)
        t.start()
        # set ready flag
        self._ready.set()
        # wait for first obs
        obs = self.obs_q.get()
        info = {}
        # return
        return obs, info

    def step(self,
             action: T_Action) -> tuple[T_Obs,
                                        T_Reward,
                                        bool,
                                        bool,
                                        dict[str, Any]]:
        assert self._ready.is_set(), 'Must call reset() first'
        # put the action
        self.action_q.put(action)
        # wait for reward
        reward = self.reward_q.get()
        # get obs
        obs = self.obs_q.get()
        # return
        return obs, reward, False, False, {}
