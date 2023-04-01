import itertools
from collections import deque
from pathlib import Path

import torch
from torch.distributions import Categorical
from torch.nn import Module
from typing_extensions import override

from mlbox.agent.basic import BasicAgent
from mlbox.agent.pg.memory import Buffer
from mlbox.trenv.queue import TerminatedError
from mlbox.types import T_Action, T_Obs


class PGAgent(BasicAgent[T_Obs, T_Action]):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = Buffer[T_Obs, T_Action]()

    #
    # props
    #

    @property
    def policy_net(self) -> Module:
        try:
            return self._policy_net
        except AttributeError:
            raise NotImplementedError('policy') from None

    @policy_net.setter
    def policy_net(self, policy_net: Module) -> None:
        self._policy_net = policy_net

    #
    # training
    #

    def policy(self,
               obs: T_Obs) -> Categorical:
        obs_tensor = torch.tensor(obs, device=self.device)
        logits = self.policy_net(obs_tensor)
        return Categorical(logits=logits)

    @override
    def learn(self) -> None:
        batch = self.buffer.get_batch()
        breakpoint()
        raise NotImplementedError()

    n_eps = 100
    batch_size = 6000
    max_step = 1000
    print_hash_every = 10
    report_progress_every = 10
    rolling_reward_ma = 5

    @override
    def train(self) -> None:
        self.policy_net.train()
        # episode loop
        rolling_reward = deque[float](maxlen=self.rolling_reward_ma)
        for i_eps in range(1, self.n_eps):
            # only train on current policy experience
            self.buffer.clear()
            # trajectory loop
            for i_batch in itertools.count():
                # reset to a new environment
                obs, *_ = self.env.reset()
                # step loop
                for _ in range(self.max_step):
                    # act
                    action = self.exploit(obs)
                    # step
                    try:
                        next_obs, reward, terminated, truncated, *_ = \
                            self.env.step(action)
                    except TerminatedError:
                        break
                    done = terminated or truncated
                    # cache experience
                    self.buffer.cache(obs, action, reward)
                    # pointing next
                    obs = next_obs
                    if done:
                        break
                # flush cache trajectory to memory
                self.buffer.flush()
                # report progress
                if i_batch % self.print_hash_every == 0:
                    print('#', end='', flush=True)
                # enough traj
                if len(self.buffer) >= self.batch_size:
                    break
            # learn from current batch
            self.learn()
            # evulate and report progress
            if i_eps % self.report_progress_every == 0:
                rolling_reward.append(self.play(self.max_step))
                mean_reward = sum(rolling_reward)/len(rolling_reward)
                print(f' | Episode {i_eps:>4d} | {mean_reward=:.1f}')

    #
    # acting
    #

    @override
    def exploit(self, obs: T_Obs) -> T_Action:
        with torch.no_grad():
            dist = self.policy(obs)
            result = dist.sample().item()
            return result

    #
    # I/O
    #

    @override
    def load(self,
             path: Path | str) -> None:
        raise NotImplementedError()

    @override
    def save(self,
             path: Path | str) -> None:
        raise NotImplementedError()
