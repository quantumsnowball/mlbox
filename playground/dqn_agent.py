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
from mlbox.utils import crop, pnl_ratio

SYMBOL = 'BTC-USD'
SYMBOLS = (SYMBOL, )
START = '2020-01-01'
END = '2020-12-31'
LENGTH = 200
INTERVAL = 5
STEP = 0.2
START_LV = 0.5
N_FEATURE = 10
MODEL_PATH = Path('model.pth')

#
# Agent
#

# what agent can observe
Obs = npt.NDArray[np.float32]
# what agent will do
Action = np.int64
# what agent will get
Reward = np.float32


class MyAgent(DQNAgent[Obs, Action, Reward]):
    device = 'cuda'
    # some normalized indicator, e.g. pnl-ratio percentage
    obs_space = Box(low=0, high=1, shape=(N_FEATURE, ), )
    in_dim = obs_space.shape[0]
    # 0 = no position, 1 = full position
    action_space = Discrete(3)
    out_dim = action_space.n.item()

    def __init__(self) -> None:
        super().__init__()
        self.policy = FullyConnected(self.in_dim, self.out_dim).to(self.device)
        self.target = FullyConnected(self.in_dim, self.out_dim).to(self.device)
        self.update_target()
        self.optimizer = torch.optim.SGD(self.policy.parameters(),
                                         lr=1e-3)
        self.loss_function = nn.CrossEntropyLoss()

    @override
    def make(self) -> Trader:
        return Trader(
            strategy=Strategy(name='Env')
            .on(SYMBOL, OhlcvWindow, do=self.explorer),
            market=YahooHistoricalWindows(
                symbols=SYMBOLS, start=START, end=END, length=LENGTH),
            broker=PaperEX(SYMBOLS)
        )

    @property
    @override
    def explorer(self) -> Hook:
        # on step, save to replay memory
        def step(my: Context[OhlcvWindow]):
            if my.count.beginning:
                # starts with half position
                my.portfolio.rebalance(SYMBOL, START_LV, my.event.price)
            elif my.count.every(INTERVAL):
                # observe
                win = my.event.win['Close']
                pnlr = pnl_ratio(win)
                feature = pnlr.iloc[-N_FEATURE:].values
                obs = np.array([feature, ], dtype=np.float32)
                # take action
                action = self.decide(obs, epilson=self.progress)
                delta_weight = +STEP * (action.item() - 1)
                target_weight = crop(my.portfolio.leverage + delta_weight,
                                     low=-1, high=+1)
                my.portfolio.rebalance(SYMBOL, target_weight, my.event.price)
                # collect experience
                eq = my.portfolio.dashboard.equity
                reward = -np.float32(eq[-1] / eq[-2] - 1)
                self.remember(obs, action, reward)

        return step


#
# model
#
agent = MyAgent()
if MODEL_PATH.is_file():
    if input(f'Model {MODEL_PATH} exists, load? (y/[n]) ').upper() == 'Y':
        # load agent
        agent.load(MODEL_PATH)
if input(f'Start training the agent? ([y]/n) ').upper() != 'N':
    # train agent
    agent.train(update_target_every=5,
                n_eps=25,
                epochs=500)
    if input(f'Save model? [y]/n) ').upper() != 'N':
        agent.save(MODEL_PATH)


#
# backtest
#
def benchmark_step(my: Context[OhlcvWindow]):
    if my.count.beginning:
        my.portfolio.rebalance(SYMBOL, 1.0, my.event.price)


def agent_step(my: Context[OhlcvWindow]):
    if my.count.beginning:
        # starts with half position
        my.portfolio.rebalance(SYMBOL, START_LV, my.event.price)
    elif my.count.every(INTERVAL):
        # observe
        win = my.event.win['Close']
        pnlr = pnl_ratio(win)
        feature = pnlr.iloc[-N_FEATURE:].values
        obs = np.array([feature, ], dtype=np.float32)
        # take action
        action = agent.exploit(obs)
        delta_weight = +STEP * (action.item() - 1)
        target_weight = crop(my.portfolio.leverage + delta_weight,
                             low=-1, high=+1)
        my.portfolio.rebalance(SYMBOL, target_weight, my.event.price)
        # mark
        my.mark['pnlr-raw'] = pnl_ratio(win)[-1]
        my.mark['action'] = action.item()
        my.mark['delta_weight'] = delta_weight
        my.mark['target_weight'] = target_weight
    my.mark['price'] = my.event.price


bt = Backtest(
    Trader(
        strategy=Strategy(name='Benchmark')
        .on(SYMBOL, OhlcvWindow, do=benchmark_step),
        market=YahooHistoricalWindows(
            symbols=SYMBOLS, start=START, end=END, length=LENGTH),
        broker=PaperEX(SYMBOLS)
    ),
    Trader(
        strategy=Strategy(name='Agent')
        .on(SYMBOL, OhlcvWindow, do=agent_step),
        market=YahooHistoricalWindows(
            symbols=SYMBOLS, start=START, end=END, length=LENGTH),
        broker=PaperEX(SYMBOLS)
    )
)
bt.run()
bt.result.save()
