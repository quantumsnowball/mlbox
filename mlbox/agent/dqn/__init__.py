from collections import deque
from inspect import currentframe
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import torch
from gymnasium import Env
from torch import float32, int64, tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing_extensions import override

from mlbox.agent import Agent
from mlbox.agent.memory import Experience, Replay
from mlbox.types import T_Action, T_Obs


class DQNAgent(Agent[T_Obs, T_Action]):
    # replay memory
    replay_size = 10000
    # learn
    n_epoch = 1000
    batch_size = 512
    gamma = 0.99
    # train
    n_eps = 100
    max_step = 10000
    skip_terminal_obs = False
    print_hash_every = 1
    update_target_every = 10
    report_progress_every = 10
    rolling_reward_ma = 5
    tracing_metrics = 'total_return'

    def __init__(self) -> None:
        super().__init__()
        self._replay = Replay[T_Obs, T_Action](self.replay_size)

    #
    # props
    #

    @property
    def policy(self) -> Module:
        try:
            return self._policy
        except AttributeError:
            raise NotImplementedError('policy') from None

    @policy.setter
    def policy(self, policy: Module) -> None:
        self._policy = policy

    @property
    def target(self) -> Module:
        try:
            return self._target
        except AttributeError:
            raise NotImplementedError('target') from None

    @target.setter
    def target(self, target: Module) -> None:
        self._target = target

    @property
    def optimizer(self) -> Optimizer:
        try:
            return self._optimizer
        except AttributeError:
            raise NotImplementedError('optimizer') from None

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    @property
    def loss_function(self) -> _Loss:
        try:
            return self._loss_function
        except AttributeError:
            raise NotImplementedError('loss_function') from None

    @loss_function.setter
    def loss_function(self, loss_function: _Loss) -> None:
        self._loss_function = loss_function

    #
    # training
    #

    def remember(self,
                 obs: T_Obs,
                 action: T_Action,
                 reward: SupportsFloat,
                 next_obs: T_Obs,
                 terminated: bool) -> None:
        self._replay.remember(
            Experience[T_Obs, T_Action](
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                terminated=terminated,
            )
        )

    def update_target(self) -> None:
        weights = self.policy.state_dict()
        self.target.load_state_dict(weights)

    @override
    def learn(self,
              n_epoch: int | None = None,
              batch_size: int | None = None,
              gamma: float | None = None) -> None:
        if n_epoch is None:
            n_epoch = self.n_epoch
        if batch_size is None:
            batch_size = self.batch_size
        if gamma is None:
            gamma = self.gamma

        for _ in range(n_epoch):
            # prepare batch of experience
            batch = self._replay.sample(batch_size, device=self.device)
            obs = batch['obs']
            action = batch['action'].unsqueeze(1)
            reward = batch['reward'].unsqueeze(1)
            next_obs = batch['next_obs']
            non_final_mask = (~batch['terminated'])
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
            # with torch.no_grad():
            #     next_sa_val = self.target(next_obs).max(1).values.unsqueeze(1)
            expected_sa_val = reward + (gamma*next_sa_val)
            # calc loss
            loss = self.loss_function(sa_val,
                                      expected_sa_val)
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # if _ == n_epoch-1:
            #     breakpoint()

    @ override
    def train(self,
              n_eps: int | None = None,
              *,
              max_step: int | None = None,
              update_target_every: int | None = None,
              report_progress_every: int | None = None,
              tracing_metrics: str | None = None,
              **kwargs: Any) -> None:
        if n_eps is None:
            n_eps = self.n_eps
        if max_step is None:
            max_step = self.max_step
        if update_target_every is None:
            update_target_every = self.update_target_every
        if report_progress_every is None:
            report_progress_every = self.report_progress_every
        if tracing_metrics is None:
            tracing_metrics = self.tracing_metrics

        self.policy.train()
        rolling_reward = deque[float](maxlen=self.rolling_reward_ma)
        for i_eps in range(1, n_eps+1):
            self.progress = min(max(i_eps/n_eps, 0), 1)
            # reset to a new environment
            obs, *_ = self.env.reset()
            # run the env
            for _ in range(max_step):
                # act
                action = self.decide(obs, epsilon=self.progress)
                # step
                next_obs, reward, terminated, truncated, *_ = \
                    self.env.step(action)
                done = terminated or truncated
                # remember
                if done and self.skip_terminal_obs:
                    break
                self.remember(obs, action, reward, next_obs, terminated)
                # pointing next
                obs = next_obs
                if done:
                    break
            # learn from experience replay
            self.learn(**kwargs)
            if i_eps % update_target_every == 0:
                self.update_target()
            # report progress
            if i_eps % self.print_hash_every == 0:
                print('#', end='', flush=True)
            if i_eps % report_progress_every == 0:
                rolling_reward.append(self.play(max_step))
                mean_reward = sum(rolling_reward)/len(rolling_reward)
                print(f' | Episode {i_eps:>4d} | {mean_reward=:.1f}')

    @override
    def play(self,
             max_step: int,
             *,
             env: Env[T_Obs, T_Action] | None = None) -> float:
        # select env
        env = env if env is not None else self.env
        # reset to a new environment
        obs, *_ = env.reset()
        # run the env
        self.policy.eval()
        total_reward = 0.0
        for _ in range(max_step):
            action = self.exploit(obs)
            next_obs, reward, terminated, *_ = env.step(action)
            obs = next_obs
            total_reward += float(reward)
            if terminated:
                break
        return total_reward

    #
    # acting
    #

    @override
    def explore(self) -> T_Action:
        random_action = self.env.action_space.sample()
        return random_action

    @override
    def exploit(self, obs: T_Obs) -> T_Action:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device)
            best_value_action = torch.argmax(self.policy(obs_tensor))
            result: T_Action = best_value_action.cpu().numpy()
            return result

    @override
    def decide(self,
               obs: T_Obs,
               *,
               epsilon: float = 0.5) -> T_Action:
        if np.random.random() > epsilon:
            return self.explore()
        else:
            return self.exploit(obs)

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

    @override
    def prompt(self,
               name: str,
               *,
               start_training: bool = False) -> None:
        # prepare caller info
        frame = currentframe()
        caller_frame = frame.f_back if frame else None
        globals = caller_frame.f_globals if caller_frame else None
        script_path = Path(globals['__file__']) if globals else Path()
        base_dir = Path(script_path).parent.relative_to(Path.cwd())
        path = base_dir / name
        if path.is_file():
            if input(f'Model {path} exists, load? (y/[n]) ').upper() == 'Y':
                # load agent
                self.load(path)
        if start_training or input(f'Start training the agent? ([y]/n) ').upper() != 'N':
            # train agent
            self.train()
            if input(f'Save model? [y]/n) ').upper() != 'N':
                self.save(path)
