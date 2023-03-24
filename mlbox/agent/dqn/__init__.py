from typing import Any, TypeVar

import numpy as np
import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from typing_extensions import override

from mlbox.agent import Agent
from mlbox.agent.memory import Replay

T_State = TypeVar('T_State')
T_Action = TypeVar('T_Action')
T_Reward = TypeVar('T_Reward')


class DQNAgent(Agent[T_State, T_Action, T_Reward]):
    def __init__(self,
                 *args: Any,
                 replay_size: int = 10000,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._replay = Replay[T_State, T_Action, T_Reward](replay_size)

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

    def update_target(self) -> None:
        weights = self.policy.state_dict()
        self.target.load_state_dict(weights)

    #
    # acting
    #

    @override
    def explore(self) -> T_Action:
        random_action = self.action_space.sample()
        return random_action

    @override
    def exploit(self, state: T_State) -> T_Action:
        state_tensor = torch.tensor(state).to(self.device)
        best_value_action = torch.argmax(self.policy(state_tensor))
        return best_value_action.cpu().numpy()

    @override
    def decide(self,
               state: T_State,
               *,
               epilson: float = 0.5) -> T_Action:
        if np.random.random() > epilson:
            return self.explore()
        else:
            return self.exploit(state)
