import numpy as np
import torch
from typing import Any, TypeVar

from torch.nn import Module

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
        self._policy: Module

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

    #
    # operations
    #

    def update_target(self) -> None:
        weights = self.policy.state_dict()
        self.target.load_state_dict(weights)

    #
    # acting
    #

    def decide(self,
               state: T_State,
               *,
               epilson: float = 0.5) -> T_Action:
        if np.random.random() > epilson:
            return self.action_space.sample()
        else:
            state_tensor = torch.tensor(state).to(self.device)
            return torch.argmax(self.policy(state_tensor)).cpu().numpy()
