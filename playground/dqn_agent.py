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
MODEL_PATH = Path('model.pth')

#
# Agent
#

# what agent can observe
State = npt.NDArray[np.float32]
# what agent will do
Action = np.int64
# what agent will get
Reward = np.float32


class MyAgent(DQNAgent[State, Action, Reward]):
    device = 'cuda'
    # 0 = no position, 1 = full position
    action_space = Discrete(3, start=-1)
    # some normalized indicator, e.g. pnl-ratio percentage
    observation_space = Box(low=0, high=1, shape=(1,), )

    def __init__(self) -> None:
        super().__init__()
        self.policy = FullyConnected(1, 2).to(self.device)
        self.target = FullyConnected(1, 2).to(self.device)
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
                feature = [pnlr, ]
                state = np.array([feature, ], dtype=np.float32)
                # take action
                action = self.decide(state, epilson=self.progress)
                delta_weight = +STEP * int(action)
                target_weight = crop(my.portfolio.leverage + delta_weight,
                                     low=-1, high=+1)
                my.portfolio.rebalance(SYMBOL, target_weight, my.event.price)
                # collect experience
                reward = np.float32(my.portfolio.leverage)
                self.remember(state, action, reward)

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
                n_eps=50,
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
        feature = [pnlr, ]
        state = np.array([feature, ], dtype=np.float32)
        # take action
        action = int(agent.exploit(state))
        delta_weight = +STEP * action
        target_weight = crop(my.portfolio.leverage + delta_weight,
                             low=-1, high=+1)
        my.portfolio.rebalance(SYMBOL, target_weight, my.event.price)
        # mark
        my.mark['pnlr-raw'] = pnl_ratio(win)
        my.mark['action'] = action
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
