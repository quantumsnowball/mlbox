import numpy as np
import torch
from gymnasium import Space
from gymnasium.spaces import Box, Discrete
from pandas import Series
from torch import nn, tensor
from trbox.broker.paper import PaperEX
from trbox.event.market import OhlcvWindow
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows
from trbox.strategy import Strategy
from trbox.strategy.context import Context
from trbox.trader import Trader
from typing_extensions import override

from mlbox.agent.dqn import DQNAgent
from mlbox.agent.memory import Experience
from mlbox.neural import FullyConnected

SYMBOL = 'BTC-USD'
SYMBOLS = (SYMBOL, )
START = '2018-01-01'
END = '2018-12-31'
LENGTH = 200

# what agent can observe
State = tuple[float, ]
# what agent will do
Action = int
# what agent will get
Reward = float


class MyAgent(DQNAgent):
    action_space = Discrete(2)
    observation_space = Box(low=0, high=1, shape=(1,))

    def __init__(self) -> None:
        super().__init__()
        self._policy = FullyConnected(1, 2).to(self._device)
        self._target = FullyConnected(1, 2).to(self._device)
        self.update_target()
        self._optimizer = torch.optim.SGD(self._policy.parameters(),
                                          lr=1e-3)
        self._loss_fn = nn.CrossEntropyLoss()

    #
    # acting
    #

    def decide(self,
               state: State,
               *,
               epilson: float = 0.5) -> Action:
        if np.random.random() > epilson:
            return self.action_space.sample()
        else:
            return int(torch.argmax(self._policy(tensor([state, ]).to(self._device))))

    #
    # training
    #

    def update_target(self) -> None:
        self._target.load_state_dict(self._policy.state_dict())

    def learn(self,
              epochs: int = 1000,
              batch_size: int = 512,
              gamma: float = 0.99):
        '''learn from reply memory'''
        for _ in range(epochs):
            batch = self._replay.sample(batch_size)
            states = torch.tensor(batch.states,
                                  dtype=torch.float32).to(self._device)
            rewards = torch.tensor(batch.rewards,
                                   dtype=torch.float32).to(self._device)
            next_states = torch.tensor(batch.next_states,
                                       dtype=torch.float32).to(self._device)
            self._policy.train()
            y = rewards + gamma*self._target(next_states)
            X = states
            pred = self._policy(X)
            loss = self._loss_fn(pred, y)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def explore(self):
        def pnl_ratio(win: Series) -> float:
            pnlr = Series(win.rank(pct=True))
            return float(pnlr[-1])

        # on step, save to replay memory
        def step(my: Context[OhlcvWindow]):
            win = my.event.win['Close']
            pnlr = pnl_ratio(win)
            state = (pnlr, )
            action = self.decide(state)
            my.portfolio.rebalance(SYMBOL, action, my.event.price)
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

    def train(self,
              n_eps: int = 1000):
        '''run trade simulation trades, save to replay, then learn'''
        for i_eps in range(n_eps):
            total_return = self.explore()
            self.learn()
            if i_eps % 50 == 0:
                self.update_target()
            print(f'total_return = {total_return:.2%} [{i_eps+1} / {n_eps}]')


agent = MyAgent()
agent.train()
