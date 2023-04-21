from typing import Generic

from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from torch.nn import Module
from torch.optim import Optimizer

from mlbox.types import T_Action, T_Obs
from mlbox.utils.wrapper import assured


class A2CProps(Generic[T_Obs, T_Action]):
    #
    # actor
    #
    @property
    @assured
    def actor_critic_net(self) -> Module:
        return self._actor_critic_net

    @actor_critic_net.setter
    def actor_critic_net(self, actor_critic_net: Module) -> None:
        self._actor_critic_net = actor_critic_net

    @property
    @assured
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer


class A2CContinuousProps(Generic[T_Obs, T_Action]):
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
        assert isinstance(self.env.action_space, Box)
        self.action_space = self.env.action_space
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.min_action = self.action_space.low
        self.max_action = self.action_space.high


class A2CDiscreteProps(Generic[T_Obs, T_Action]):
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
        self.action_dim = self.env.action_space.n
