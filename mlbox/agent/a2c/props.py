from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from torch.nn import Module
from torch.optim import Optimizer

from mlbox.interface.agent import Agent
from mlbox.types import T_Action, T_Obs
from mlbox.utils.wrapper import assured


class Props(Agent[T_Obs, T_Action]):
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
