from pathlib import Path

import numpy as np
import numpy.typing as npt
from trbox.broker.paper import PaperEX
from trbox.event.market import OhlcvWindow
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows
from trbox.strategy import Context, Strategy
from trbox.trader import Trader
from typing_extensions import override

from mlbox.trenv import TrEnv
from mlbox.utils import crop

SYMBOL = 'BTC-USD'
SYMBOLS = (SYMBOL, )
START = '2020-01-01'
END = '2020-12-31'
LENGTH = 200
INTERVAL = 5
STEP = 0.2
START_LV = 0.01
N_FEATURE = LENGTH-1
MODEL_PATH = Path('model.pth')

Obs = npt.NDArray[np.float32]
Action = np.int64
Reward = np.float32


class Env(TrEnv):
    @override
    def observe(self, my: Context[OhlcvWindow]) -> Obs:
        win = my.event.win['Close']
        pct_chg = win.pct_change().dropna()
        feature = np.array(pct_chg[-N_FEATURE:])
        obs = np.array([feature, ], dtype=np.float32)
        return obs

    @override
    def act(self, my: Context[OhlcvWindow], action: Action) -> None:
        delta_weight = +STEP * (action.item() - 1)
        target_weight = crop(my.portfolio.leverage + delta_weight,
                             low=-1, high=+1)
        my.portfolio.rebalance(SYMBOL, target_weight, my.event.price)

    @override
    def grant(self, my: Context[OhlcvWindow]) -> Reward:
        eq = my.portfolio.dashboard.equity
        pr = my.memory['price'][INTERVAL]
        eq_r = np.float32(eq[-1] / eq[-INTERVAL] - 1)
        pr_r = np.float32(pr[-1] / pr[-INTERVAL] - 1)
        reward = eq_r - pr_r
        return reward

    @override
    def make(self) -> Trader:
        return Trader(
            strategy=Strategy(name='TrEnv')
            .on(SYMBOL, OhlcvWindow, do=self.do),
            market=YahooHistoricalWindows(
                symbols=SYMBOLS, start=START, end=END, length=LENGTH),
            broker=PaperEX(SYMBOLS)
        )


env = Env()
env.reset()
