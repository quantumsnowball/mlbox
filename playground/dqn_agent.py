import random
from collections import deque
from dataclasses import astuple, dataclass

import numpy as np
import torch
from pandas import Series, Timestamp, to_datetime
from torch import Tensor, nn, tensor
from trbox.broker.paper import PaperEX
from trbox.common.types import Symbol
from trbox.event.market import OhlcvWindow
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows
from trbox.strategy import Strategy
from trbox.strategy.context import Context
from trbox.trader import Trader

State = float
Action = int


@dataclass
class Experience:
    state: State
    action: int
    reward: float
    next_state: State
    done: bool


class Replay:
    def __init__(self, maxlen):
        self._memory = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._memory)

    def remember(self, exp: Experience):
        state, action, reward, next_state, done = astuple(exp)
        self._memory.append((state, action, reward, next_state, done))

    def get_batch(self, batch_size):
        samples = random.sample(self._memory, min(
            len(self._memory), batch_size))
        batch = np.array(samples).transpose()
        states, actions, rewards, next_states, dones = batch
        states, next_states = np.stack(states), np.stack(next_states)
        return states, actions, rewards, next_states, dones


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x: Tensor):
        logits = self.network(x)
        return logits


class Agent:
    def __init__(self,
                 symbol: Symbol,
                 start: Timestamp | str,
                 end: Timestamp | str,
                 length: int) -> None:
        self._device = 'cuda'
        self._symbol = symbol
        self._symbols = (symbol, )
        self._start = to_datetime(start)
        self._end = to_datetime(end)
        self._length = length
        self._replay = Replay(10000)
        self._policy = Policy().to(self._device)
        self._target = Policy().to(self._device)
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
            return int(np.random.choice([0, 1]))
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
        for i_eps in range(epochs):
            states, actions, rewards, next_states, _ = \
                self._replay.get_batch(batch_size)
            states = torch.tensor(states,
                                  dtype=torch.float32).reshape(-1, 1).to(self._device)
            rewards = torch.tensor(rewards,
                                   dtype=torch.float32).reshape(-1, 1).to(self._device)
            next_states = torch.tensor(next_states,
                                       dtype=torch.float32).reshape(-1, 1).to(self._device)
            # actions = torch.tensor(actions, dtype=torch.int32)
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
            state = pnl_ratio(win)
            action = self.decide(state)
            my.portfolio.rebalance(self._symbol, action, my.event.price)
            my.memory['state'][2].append(state)
            my.memory['action'][2].append(action)
            try:
                exp = Experience(
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
            .on(self._symbol, OhlcvWindow, do=step),
            market=YahooHistoricalWindows(
                symbols=self._symbols,
                start=self._start,
                end=self._end,
                length=self._length),
            broker=PaperEX(self._symbols)
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


agent = Agent('BTC-USD', '2018-01-01', '2018-12-31', 200)
agent.train()
