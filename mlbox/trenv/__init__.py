from abc import ABC, abstractmethod
from threading import Event, Thread
from typing import Any, Self, SupportsFloat

from gymnasium import Env
from pandas import Timestamp
from trbox.broker.paper import PaperEX
from trbox.common.logger import Log
from trbox.common.types import Symbol
from trbox.event.market import OhlcvWindow
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows
from trbox.strategy.context import Context
from trbox.strategy.types import Hook
from trbox.trader import Trader

from mlbox.events import TerminatedError
from mlbox.trenv.queue import TrEnvQueue
from mlbox.trenv.strategy import TrEnvStrategy
from mlbox.types import T_Action, T_Obs


class TrEnv(Env[T_Obs, T_Action], ABC):
    Market: type[YahooHistoricalWindows]
    interval: int
    symbol: Symbol
    start: Timestamp | str
    end: Timestamp | str
    length: int

    def __new__(cls) -> Self:
        try:
            # ensure attrs are implemented in subclass instance
            cls.observation_space
            cls.action_space
            cls.Market
            cls.interval
            cls.symbol
            cls.start
            cls.end
            cls.length
            return super().__new__(cls)
        except AttributeError as e:
            raise NotImplementedError(e.name) from None

    def __init__(self) -> None:
        super().__init__()
        self.obs_q = TrEnvQueue[T_Obs]()
        self.action_q = TrEnvQueue[T_Action]()
        self.reward_q = TrEnvQueue[SupportsFloat]()
        self._ready = Event()
        self._trader: Trader

    @abstractmethod
    def observe(self, my: Context[OhlcvWindow]) -> T_Obs:
        ...

    @abstractmethod
    def act(self, my: Context[OhlcvWindow], action: T_Action) -> None:
        ...

    @abstractmethod
    def grant(self, my: Context[OhlcvWindow]) -> SupportsFloat:
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

    def make(self) -> Trader:
        return Trader(
            strategy=TrEnvStrategy[T_Obs, T_Action](name='TrEnv', trenv=self)
            .on(self.symbol, OhlcvWindow, do=self.do),
            market=self.Market(
                symbols=(self.symbol,),
                start=self.start,
                end=self.end,
                length=self.length),
            broker=PaperEX((self.symbol,))
        )

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
        # reset queues
        self.obs_q.clear()
        self.action_q.clear()
        self.reward_q.clear()
        # create env
        self._trader = self.make()
        t = Thread(target=self._trader.run, daemon=True)
        t.start()
        # wait for first obs
        Log.info('waiting for obs')
        obs = self.obs_q.get()
        Log.info('got obs')
        # set ready flag
        self._ready.set()
        # return
        return obs, {}

    def step(self,
             action: T_Action) -> tuple[T_Obs,
                                        SupportsFloat,
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
        except TerminatedError as e:
            # reset flag
            self._ready.clear()
            # agent should handle this
            raise e
