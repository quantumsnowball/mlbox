from abc import ABC, abstractmethod
from threading import Event, Thread
from typing import Any, Generic, Self, SupportsFloat, TypeVar

import numpy as np
from gymnasium import Env
from trbox.common.logger import Log
from trbox.event.market import OhlcvWindow
from trbox.strategy import Context, Hook
from trbox.trader import Trader

from mlbox.trenv.queue import TerminatedError, TrEnvQueue

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
        self.obs_q = TrEnvQueue[T_Obs]()
        self.action_q = TrEnvQueue[T_Action]()
        self.reward_q = TrEnvQueue[T_Reward]()
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
                try:
                    # take action
                    Log.info('waiting for action')
                    action = self.action_q.get()  # blocking
                    Log.info('got action')
                    self.act(my, action)
                except TerminatedError:
                    # stop waiting for further action
                    return
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
        # wait for first obs
        Log.info('waiting for obs')
        obs = self.obs_q.get()
        Log.info('got obs')
        info = {}
        # set ready flag
        self._ready.set()
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
        try:
            # wait for reward
            Log.info('waiting for reward')
            reward = self.reward_q.get()  # blocking
            Log.info('got reward')
            # get obs
            Log.info('waiting for obs')
            obs = self.obs_q.get()  # blocking
            Log.info('got obs')
            # return
            return obs, reward, False, False, {}
        except TerminatedError:
            # # return dummy
            obs: Any = np.array([[]])
            reward: Any = 0.0
            return obs, reward, True, False, {}
