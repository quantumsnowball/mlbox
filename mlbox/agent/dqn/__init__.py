from inspect import currentframe
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import torch
from torch import float32, tensor
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
    update_target_every = 10
    report_progress_every = 1
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
            batch = self._replay.sample(batch_size)
            obs = tensor(batch.obs,
                         dtype=float32).to(self.device)
            reward = tensor(batch.reward,
                            dtype=float32).to(self.device)
            next_obs = tensor(batch.next_obs,
                              dtype=float32).to(self.device)
            # train mode
            self.policy.train()
            # calc features and targets
            X = obs
            y = reward + gamma*self.target(next_obs)
            # back propagation
            predicted = self.policy(X)
            loss = self.loss_function(predicted, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self,
              n_eps: int | None = None,
              *,
              update_target_every: int | None = None,
              report_progress_every: int | None = None,
              tracing_metrics: str | None = None,
              **kwargs: Any) -> None:
        if n_eps is None:
            n_eps = self.n_eps
        if update_target_every is None:
            update_target_every = self.update_target_every
        if report_progress_every is None:
            report_progress_every = self.report_progress_every
        if tracing_metrics is None:
            tracing_metrics = self.tracing_metrics

        for i_eps in range(n_eps):
            self.progress = min(max(i_eps/n_eps, 0), 1)
            # reset to a new environment
            obs, *_ = self.env.reset()
            # run the env
            cum_reward = 0.0
            while True:
                action = self.decide(obs, epsilon=self.progress)
                next_obs, reward, terminated, *_ = self.env.step(action)
                if terminated:
                    break
                self.remember(obs, action, reward, next_obs, terminated)
                obs = next_obs
                cum_reward += float(reward)
            # learn from experience replay
            self.learn(**kwargs)
            # result = getattr(self.env.portfolio.metrics,
            #                  tracing_metrics, float('nan'))
            if i_eps % update_target_every == 0:
                self.update_target()
            if i_eps % report_progress_every == 0:
                print(
                    f'[{i_eps+1: >3} / {n_eps}] '
                    f'cum_reward = {cum_reward:+.4f}'
                )

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
            obs_tensor = torch.tensor(obs).to(self.device)
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
               name: str) -> None:
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
        if input(f'Start training the agent? ([y]/n) ').upper() != 'N':
            # train agent
            self.train()
            if input(f'Save model? [y]/n) ').upper() != 'N':
                self.save(path)
