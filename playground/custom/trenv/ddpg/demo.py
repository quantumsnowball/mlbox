import numpy as np
import numpy.typing as npt
import torch as T
import torch.optim as optim
from gymnasium.spaces import Box
from trbox.backtest import Backtest
from trbox.broker.paper import PaperEX
from trbox.event.market import OhlcvWindow
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows
from trbox.strategy import Strategy
from trbox.strategy.context import Context
from trbox.strategy.types import Hook
from trbox.trader import Trader
from typing_extensions import override

from mlbox.agent.ddpg import DDPGAgent
from mlbox.agent.ddpg.nn import DDPGActorNet, DDPGCriticNet
from mlbox.trenv import TrEnv
from mlbox.utils import crop, pnl_ratio

SYMBOL = 'BTC-USD'
SYMBOLS = (SYMBOL, )
START = '2022-01-01'
END = '2022-12-31'
LENGTH = 200
INTERVAL = 5
STEP = 0.2
START_LV = 0.01
N_FEATURE = 150
MODEL_NAME = 'model.pth'

Obs = npt.NDArray[np.float32]
Action = np.float32
Reward = np.float32


#
# routine
#
def observe(my: Context[OhlcvWindow]) -> Obs:
    win = my.event.win['Close']
    pnlr = pnl_ratio(win)
    obs = np.array(pnlr[-N_FEATURE:], dtype=np.float32)
    return obs


def act(my: Context[OhlcvWindow], action: Action) -> float:
    target_weight = crop(action.item(), low=0, high=1)
    my.portfolio.rebalance(SYMBOL, target_weight, my.event.price)
    return target_weight


def grant(my: Context[OhlcvWindow]) -> Reward:
    eq = my.portfolio.dashboard.equity
    # pr = my.memory['price'][INTERVAL]
    eq_r = np.float32(eq[-1] / eq[-INTERVAL] - 1)
    # pr_r = np.float32(pr[-1] / pr[-INTERVAL] - 1)
    reward = eq_r  # - pr_r
    return reward


def every(my: Context[OhlcvWindow]) -> None:
    my.memory['price'][INTERVAL].append(my.event.price)


#
# Env
#
class MyEnv(TrEnv[Obs, Action]):
    # Env
    observation_space: Box = Box(low=0, high=1, shape=(N_FEATURE, ), )
    action_space: Box = Box(low=0, high=1, shape=(1, ), )

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
class MyAgent(DDPGAgent[Obs, Action]):
    device = T.device('cuda')
    max_step = 500
    n_eps = 5000
    n_epoch = 3
    replay_size = 1000*max_step
    batch_size = 128
    update_target_every = 10
    print_hash_every = 5
    rolling_reward_ma = 20
    report_progress_every = 50
    render_every = 500
    mean_reward_display_format = '+.4'

    def __init__(self) -> None:
        super().__init__()
        self.env = MyEnv()
        self.render_env = self.env
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Box)
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        high = self.env.action_space.high
        low = self.env.action_space.low
        self.min_noise = 0.2
        self.max_noise = high * 5
        self.actor_net = DDPGActorNet(obs_dim, action_dim,
                                      min_action=low,
                                      max_action=high).to(self.device)
        self.actor_net_target = DDPGActorNet(obs_dim, action_dim,
                                             min_action=low,
                                             max_action=high).to(self.device)
        self.critic_net = DDPGCriticNet(obs_dim, action_dim).to(self.device)
        self.critic_net_target = DDPGCriticNet(obs_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=1e-3)


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
        target_weight = act(my, action)
        # mark
        # my.mark['pnlr-raw'] = pnl_ratio(win)[-1]
        my.mark['action'] = action.item()
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
agent.prompt(MODEL_NAME)
backtest.run()
backtest.result.save()
