from collections import deque
from pathlib import Path

import torch
from torch.nn import Module
from typing_extensions import override

from mlbox.agent.basic import BasicAgent
from mlbox.agent.dqn.memory import CachedReplay
from mlbox.trenv.queue import TerminatedError
from mlbox.types import T_Action, T_Obs
from mlbox.utils.wrapper import assured


class DQNAgent(BasicAgent[T_Obs, T_Action]):
    # replay memory
    replay_size = 10000

    def __init__(self) -> None:
        super().__init__()
        self.replay = CachedReplay[T_Obs, T_Action](self.replay_size)

    #
    # props
    #

    @property
    @assured
    def policy(self) -> Module:
        return self._policy

    @policy.setter
    def policy(self, policy: Module) -> None:
        self._policy = policy

    @property
    @assured
    def target(self) -> Module:
        return self._target

    @target.setter
    def target(self, target: Module) -> None:
        self._target = target

    #
    # training
    #
    def update_target(self) -> None:
        weights = self.policy.state_dict()
        self.target.load_state_dict(weights)

    n_epoch = 1000
    batch_size = 512
    gamma = 0.99

    @override
    def learn(self) -> None:
        for _ in range(self.n_epoch):
            # prepare batch of experience
            batch = self.replay.sample(self.batch_size, device=self.device)
            obs = batch.obs
            action = batch.action.unsqueeze(1)
            reward = batch.reward.unsqueeze(1)
            next_obs = batch.next_obs
            non_final_mask = (~batch.terminated)
            non_final_next_obs = next_obs[non_final_mask]
            # set train mode
            self.policy.train()
            # calc state-action value
            sa_val = self.policy(obs).gather(1, action)
            # calc expected state-action value
            next_sa_val = torch.zeros(len(obs), 1, device=self.device)
            with torch.no_grad():
                next_sa_val[non_final_mask] = self.target(
                    non_final_next_obs).max(1).values.unsqueeze(1)
            expected_sa_val = reward + self.gamma*next_sa_val
            # calc loss
            loss = self.loss_function(sa_val,
                                      expected_sa_val)
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # if _ == n_epoch-1:
            #     breakpoint()

    n_eps = 100
    max_step = 10000
    update_target_every = 10
    report_progress_every = 10
    print_hash_every = 1
    rolling_reward_ma = 5

    @ override
    def train(self) -> None:
        self.policy.train()
        rolling_reward = deque[float](maxlen=self.rolling_reward_ma)
        for i_eps in range(1, self.n_eps+1):
            self.progress = min(max(i_eps/self.n_eps, 0), 1)
            # reset to a new environment
            obs, *_ = self.env.reset()
            # run the env
            for _ in range(self.max_step):
                # act
                action = self.decide(obs, epsilon=self.progress)
                # step
                try:
                    next_obs, reward, terminated, truncated, *_ = \
                        self.env.step(action)
                except TerminatedError:
                    break
                done = terminated or truncated
                # cache experience
                self.replay.cache(obs, action, reward, next_obs, terminated)
                # pointing next
                obs = next_obs
                if done:
                    break
            # post processing to cached experience before flush
            self.replay.assert_terminated_flag()
            # flush cache experience to memory
            self.replay.flush()
            # learn from experience replay
            self.learn()
            if i_eps % self.update_target_every == 0:
                self.update_target()
            # report progress
            if i_eps % self.print_hash_every == 0:
                print('#', end='', flush=True)
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
            obs_tensor = torch.tensor(obs, device=self.device)
            best_value_action = torch.argmax(self.policy(obs_tensor))
            result: T_Action = best_value_action.cpu().numpy()
            return result

    #
    # I/O
    #

    @override
    def load(self,
             path: Path | str) -> None:
        path = Path(path)
        self.policy.load_state_dict(torch.load(path))
        self.update_target()
        print(f'Loaded model: {path}')

    @override
    def save(self,
             path: Path | str) -> None:
        path = Path(path)
        torch.save(self.policy.state_dict(), path)
        print(f'Saved model: {path}')
