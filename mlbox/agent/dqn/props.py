from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from mlbox.interface.agent import Agent
from mlbox.types import T_Action, T_Obs
from mlbox.utils.wrapper import assured


class Props(Agent[T_Obs, T_Action]):
    #
    # env
    #

    @property
    @assured
    def env(self) -> Env[T_Obs, T_Action]:
        return self._env

    @env.setter
    def env(self, env: Env[T_Obs, T_Action]) -> None:
        self._env = env
        assert isinstance(self.env.observation_space, Box)
        assert isinstance(self.env.action_space, Discrete)
        self.action_space = self.env.action_space
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n.item()

    #
    # agent
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

    @property
    @assured
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    @property
    @assured
    def loss_function(self) -> _Loss:
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function: _Loss) -> None:
        self._loss_function = loss_function
