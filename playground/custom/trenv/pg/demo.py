import numpy as np
import numpy.typing as npt
import torch as T
from gymnasium.spaces import Box, Discrete
from torch.nn import ReLU, Tanh
from torch.optim import Adam
from trbox.backtest import Backtest
from trbox.broker.paper import PaperEX
from trbox.event.market import OhlcvWindow
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows
from trbox.strategy import Strategy
from trbox.strategy.context import Context
from trbox.strategy.types import Hook
from trbox.trader import Trader
from typing_extensions import override

from mlbox.agent.pg import PGAgent
from mlbox.agent.pg.nn import BaselineNet, PolicyNet
from mlbox.trenv import TrEnv
from mlbox.utils import crop, pnl_ratio

SYMBOL = 'BTC-USD'
SYMBOLS = (SYMBOL, )
START = '2020-01-01'
END = '2020-12-31'
LENGTH = 200
INTERVAL = 5
STEP = 0.2
START_LV = 0.01
N_FEATURE = 150
MODEL_NAME = 'model.pth'

Obs = npt.NDArray[np.float32]
Action = npt.NDArray[np.int64]
Reward = np.float32


#
# routine
#
def observe(my: Context[OhlcvWindow]) -> Obs:
    win = my.event.win['Close']
    pnlr = pnl_ratio(win)
    obs = np.array(pnlr[-N_FEATURE:], dtype=np.float32)
    return obs


def act(my: Context[OhlcvWindow], action: Action) -> tuple[float, float]:
    delta_weight = +STEP * (action.item() - 1)
    target_weight = crop(my.portfolio.leverage + delta_weight,
                         low=-1, high=+1)
    my.portfolio.rebalance(SYMBOL, target_weight, my.event.price)
    return delta_weight, target_weight


def grant(my: Context[OhlcvWindow]) -> Reward:
    eq = my.portfolio.dashboard.equity
    pr = my.memory['price'][INTERVAL]
    eq_r = np.float32(eq[-1] / eq[-INTERVAL] - 1)
    pr_r = np.float32(pr[-1] / pr[-INTERVAL] - 1)
    reward = eq_r - pr_r
    return reward


def every(my: Context[OhlcvWindow]) -> None:
    my.memory['price'][INTERVAL].append(my.event.price)


#
# Env
#
class MyEnv(TrEnv[Obs, Action]):
    # Env
    observation_space: Box = Box(low=0, high=1, shape=(N_FEATURE, ), )
    action_space: Discrete = Discrete(3)

    # Trader
    Market = YahooHistoricalWindows
    interval = INTERVAL
    symbol = SYMBOL
    start = START
    end = END
    length = LENGTH

    @override
    def observe(self, my: Context[OhlcvWindow]) -> Obs:
        return observe(my)

    @override
    def act(self, my: Context[OhlcvWindow], action: Action) -> None:
        act(my, action)

    @override
    def grant(self, my: Context[OhlcvWindow]) -> Reward:
        return grant(my)

    @override
    def every(self, my: Context[OhlcvWindow]) -> None:
        every(my)


#
# Agent
#
class MyAgent(PGAgent[Obs, Action]):
    device = T.device('cpu')
    max_step = 500
    n_eps = 50
    batch_size = 1250
    report_progress_every = 1
    reward_to_go = True
    baseline = True

    def __init__(self) -> None:
        super().__init__()
        self.env = MyEnv()
        self.policy_net = PolicyNet(self.obs_dim, self.action_dim,
                                    hidden_dim=32,
                                    Activation=Tanh).to(self.device)
        self.baseline_net = BaselineNet(self.obs_dim, 1,
                                        hidden_dim=32,
                                        Activation=ReLU).to(self.device)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=1e-2)
        self.baseline_optimizer = Adam(self.policy_net.parameters(), lr=1e-3)


#
# backtest
#
def benchmark_step(my: Context[OhlcvWindow]):
    if my.count.beginning:
        my.portfolio.rebalance(SYMBOL, 1.0, my.event.price)


def agent_step(my: Context[OhlcvWindow]):
    every(my)
    if my.count.beginning:
        # starts with half position
        my.portfolio.rebalance(SYMBOL, START_LV, my.event.price)
    elif my.count.every(INTERVAL):
        # observe
        obs = observe(my)
        # take action
        action = agent.decide(obs)
        delta_weight, target_weight = act(my, action)
        # mark
        # my.mark['pnlr-raw'] = pnl_ratio(win)[-1]
        my.mark['action'] = action.item()
        my.mark['delta_weight'] = delta_weight
        my.mark['target_weight'] = target_weight
        my.mark['reward'] = grant(my)
        my.mark['cum_reward'] = sum(my.mark['reward'])
    my.mark['price'] = my.event.price


def Env(name: str, do: Hook[OhlcvWindow]) -> Trader:
    return Trader(
        strategy=Strategy(name=name)
        .on(SYMBOL, OhlcvWindow, do=do),
        market=YahooHistoricalWindows(
            symbols=SYMBOLS, start=START, end=END, length=LENGTH),
        broker=PaperEX(SYMBOLS)
    )


backtest = Backtest(
    Env('Benchmark', benchmark_step),
    Env('Agent', agent_step)
)

#
# main
#
agent = MyAgent()
agent.prompt(MODEL_NAME, start_training=True)
backtest.run()
backtest.result.save()
