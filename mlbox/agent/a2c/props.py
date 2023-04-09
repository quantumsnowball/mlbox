from torch.nn import Module
from torch.optim import Optimizer

from mlbox.utils.wrapper import assured


class A2CProps:
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
