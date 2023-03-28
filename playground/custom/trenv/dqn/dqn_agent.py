from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn
from trbox.backtest import Backtest
from trbox.broker.paper import PaperEX
from trbox.event.market import OhlcvWindow
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows
from trbox.strategy import Hook, Strategy
from trbox.strategy.context import Context
from trbox.trader import Trader
from typing_extensions import override

from mlbox.agent.dqn import DQNAgent
from mlbox.neural import FullyConnected
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


def Env(name: str, do: Hook[OhlcvWindow]) -> Trader:
    return Trader(
        strategy=Strategy(name=name)
        .on(SYMBOL, OhlcvWindow, do=do),
        market=YahooHistoricalWindows(
            symbols=SYMBOLS, start=START, end=END, length=LENGTH),
        broker=PaperEX(SYMBOLS)
    )


def observe(my: Context[OhlcvWindow]) -> Obs:
    win = my.event.win['Close']
    pct_chg = win.pct_change().dropna()
    # pnlr = pnl_ratio(win)
    # feature = pnlr.iloc[-N_FEATURE:].values
    feature = np.array(pct_chg[-N_FEATURE:])
    obs = np.array([feature, ], dtype=np.float32)
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


#
# Agent
#

class MyAgent(DQNAgent[Obs, Action, Reward]):
    device = 'cuda'
    replay_size = 100
    update_target_every = 5
    n_eps = 40
    n_epoch = 500
    gamma = 1.0

    # obs
    obs_space = Box(low=0, high=1, shape=(N_FEATURE, ), )
    in_dim = obs_space.shape[0]

    # action
    action_space = Discrete(3)
    out_dim = action_space.n.item()

    def __init__(self) -> None:
        super().__init__()
        self.policy = FullyConnected(self.in_dim, self.out_dim).to(self.device)
        self.target = FullyConnected(self.in_dim, self.out_dim).to(self.device)
        self.update_target()
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=1e-3)
        self.loss_function = nn.CrossEntropyLoss()

    @override
    def make(self) -> Trader:
        return Env('Env', self.explorer)

    @property
    @override
    def explorer(self) -> Hook:
        # on step, save to replay memory
        def step(my: Context[OhlcvWindow]):
            my.memory['price'][INTERVAL].append(my.event.price)
            if my.count.beginning:
                # starts with half position
                my.portfolio.rebalance(SYMBOL, START_LV, my.event.price)
            elif my.count.every(INTERVAL):
                # observe
                obs = observe(my)
                # take action
                action = self.decide(obs, epsilon=self.progress)
                act(my, action)
                # collect experience
                reward = grant(my)
                self.remember(obs, action, reward)

        return step


#
# model
#
agent = MyAgent()
agent.prompt(MODEL_PATH)


#
# backtest
#
def benchmark_step(my: Context[OhlcvWindow]):
    if my.count.beginning:
        my.portfolio.rebalance(SYMBOL, 1.0, my.event.price)


def agent_step(my: Context[OhlcvWindow]):
    my.memory['price'][INTERVAL].append(my.event.price)
    if my.count.beginning:
        # starts with half position
        my.portfolio.rebalance(SYMBOL, START_LV, my.event.price)
    elif my.count.every(INTERVAL):
        # observe
        obs = observe(my)
        # take action
        action = agent.exploit(obs)
        delta_weight, target_weight = act(my, action)
        # mark
        # my.mark['pnlr-raw'] = pnl_ratio(win)[-1]
        my.mark['action'] = action.item()
        my.mark['delta_weight'] = delta_weight
        my.mark['target_weight'] = target_weight
        my.mark['reward'] = grant(my)
        my.mark['cum_reward'] = sum(my.mark['reward'])
    my.mark['price'] = my.event.price


bt = Backtest(
    Env('Benchmark', benchmark_step),
    Env('Agent', agent_step)
)
bt.run()
bt.result.save()
