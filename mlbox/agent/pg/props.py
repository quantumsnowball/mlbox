from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from torch.nn import Module
from torch.optim import Optimizer

from mlbox.interface.agent import Agent
from mlbox.types import T_Action, T_Obs
from mlbox.utils.wrapper import assured


class PGProps(Agent[T_Obs, T_Action]):
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
        self.observation_space: Box = self.env.observation_space
        self.action_space: Discrete = self.env.action_space
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.n.item()

    #
    # agent
    #
    @property
    @assured
    def policy_net(self) -> Module:
        return self._policy_net

    @policy_net.setter
    def policy_net(self, policy_net: Module) -> None:
        self._policy_net = policy_net

    @property
    @assured
    def baseline_net(self) -> Module:
        return self._baseline_net

    @baseline_net.setter
    def baseline_net(self, baseline_net: Module) -> None:
        self._baseline_net = baseline_net

    @property
    @assured
    def policy_optimizer(self) -> Optimizer:
        return self._policy_optimizer

    @policy_optimizer.setter
    def policy_optimizer(self, policy_optimizer: Optimizer) -> None:
        self._policy_optimizer = policy_optimizer

    @property
    @assured
    def baseline_optimizer(self) -> Optimizer:
        return self._baseline_optimizer

    @baseline_optimizer.setter
    def baseline_optimizer(self, baseline_optimizer: Optimizer) -> None:
        self._baseline_optimizer = baseline_optimizer
