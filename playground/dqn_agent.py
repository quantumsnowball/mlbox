import numpy as np
import numpy.typing as npt
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn
from trbox.broker.paper import PaperEX
from trbox.event.market import OhlcvWindow
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows
from trbox.strategy import Strategy
from trbox.strategy.context import Context
from trbox.trader import Trader

from mlbox.agent.dqn import DQNAgent
from mlbox.agent.memory import Experience
from mlbox.neural import FullyConnected
from mlbox.utils import pnl_ratio

SYMBOL = 'BTC-USD'
SYMBOLS = (SYMBOL, )
START = '2018-01-01'
END = '2018-12-31'
LENGTH = 200


# what agent can observe
State = npt.NDArray[np.float32]
# what agent will do
Action = np.int64
# what agent will get
Reward = np.float32


class MyAgent(DQNAgent[State, Action, Reward]):
    device = 'cuda'
    # 0 = no position, 1 = full position
    action_space = Discrete(2)
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

    #
    # training
    #

    def research(self) -> float:
        # on step, save to replay memory
        def step(my: Context[OhlcvWindow]):
            win = my.event.win['Close']
            pnlr = pnl_ratio(win)
            feature = [pnlr, ]
            state = np.array([feature, ], dtype=np.float32)
            action = self.decide(state)
            my.portfolio.rebalance(SYMBOL, float(action), my.event.price)
            my.memory['state'][2].append(state)
            my.memory['action'][2].append(action)
            try:
                exp = Experience[State, Action, Reward](
                    state=my.memory['state'][2][-2],
                    action=my.memory['action'][2][-2],
                    reward=my.dashboard.equity[-1]/my.dashboard.equity[-2] - 1,
                    next_state=my.memory['state'][2][-1],
                    done=False,
                )
                self._replay.remember(exp)
            except IndexError:
                pass

        # run simulation
        t = Trader(
            strategy=Strategy(name='agent')
            .on(SYMBOL, OhlcvWindow, do=step),
            market=YahooHistoricalWindows(
                symbols=SYMBOLS,
                start=START,
                end=END,
                length=LENGTH),
            broker=PaperEX(SYMBOLS)
        )
        t.run()
        return t.portfolio.metrics.total_return


agent = MyAgent()
agent.train()
