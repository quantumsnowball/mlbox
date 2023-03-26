from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
from torch import float32, tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing_extensions import override

from mlbox.agent import Agent
from mlbox.agent.memory import Experience, Replay

T_Obs = TypeVar('T_Obs')
T_Action = TypeVar('T_Action')
T_Reward = TypeVar('T_Reward')


class DQNAgent(Agent[T_Obs, T_Action, T_Reward]):
    def __init__(self,
                 *args: Any,
                 replay_size: int = 10000,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._replay = Replay[T_Obs, T_Action, T_Reward](replay_size)

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
                 reward: T_Reward) -> None:
        # remember value if lag-1 exists
        try:
            self._replay.remember(
                Experience[T_Obs, T_Action, T_Reward](
                    # lag-1 values
                    obs=self._prev_obs,
                    action=self._prev_action,
                    # current values
                    reward=reward,
                    next_obs=obs,
                    done=False,
                )
            )
        except AttributeError:
            pass

        # update lag-1 values
        self._prev_obs = obs
        self._prev_action = action

    def update_target(self) -> None:
        weights = self.policy.state_dict()
        self.target.load_state_dict(weights)

    @override
    def learn(self,
              epochs: int = 1000,
              batch_size: int = 512,
              gamma: float = 0.99) -> None:
        for _ in range(epochs):
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
              n_eps: int = 100,
              *,
              update_target_every: int = 10,
              report_progress_every: int = 1,
              tracing: str = 'total_return',
              **kwargs: Any) -> None:
        for i_eps in range(n_eps):
            self.progress = min(max(i_eps/n_eps, 0), 1)
            # create a new environment
            self.reset()
            # run the env
            self.env.run()
            # learn from experience replay
            self.learn(**kwargs)
            result = getattr(self.env.portfolio.metrics, tracing, float('nan'))
            if i_eps % update_target_every == 0:
                self.update_target()
            if i_eps % report_progress_every == 0:
                print(
                    f'{tracing} = {result:+.4f} '
                    f'[{i_eps+1} / {n_eps}]'
                )

    #
    # acting
    #

    @override
    def explore(self) -> T_Action:
        random_action = self.action_space.sample()
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
               epilson: float = 0.5) -> T_Action:
        if np.random.random() > epilson:
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
               path: Path | str) -> None:
        path = Path(path)
        if path.is_file():
            if input(f'Model {path} exists, load? (y/[n]) ').upper() == 'Y':
                # load agent
                self.load(path)
        if input(f'Start training the agent? ([y]/n) ').upper() != 'N':
            # train agent
            self.train(update_target_every=5,
                       n_eps=25,
                       epochs=500)
            if input(f'Save model? [y]/n) ').upper() != 'N':
                self.save(path)
