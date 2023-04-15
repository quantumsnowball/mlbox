from torch.nn import Module
from torch.optim import Optimizer

from mlbox.utils.wrapper import assured


class DDPGProps:
    #
    # actor
    #
    @property
    @assured
    def actor_net(self) -> Module:
        return self._actor_net

    @actor_net.setter
    def actor_net(self, actor_net: Module) -> None:
        self._actor_net = actor_net

    @property
    @assured
    def actor_net_target(self) -> Module:
        return self._actor_net_target

    @actor_net_target.setter
    def actor_net_target(self, actor_net_target: Module) -> None:
        self._actor_net_target = actor_net_target

    @property
    @assured
    def actor_optimizer(self) -> Optimizer:
        return self._actor_optimizer

    @actor_optimizer.setter
    def actor_optimizer(self, actor_optimizer: Optimizer) -> None:
        self._actor_optimizer = actor_optimizer

    #
    # critic
    #
    @property
    @assured
    def critic_net(self) -> Module:
        return self._critic_net

    @critic_net.setter
    def critic_net(self, critic_net: Module) -> None:
        self._critic_net = critic_net

    @property
    @assured
    def critic_net_target(self) -> Module:
        return self._critic_net_target

    @critic_net_target.setter
    def critic_net_target(self, critic_net_target: Module) -> None:
        self._critic_net_target = critic_net_target

    @property
    @assured
    def critic_optimizer(self) -> Optimizer:
        return self._critic_optimizer

    @critic_optimizer.setter
    def critic_optimizer(self, critic_optimizer: Optimizer) -> None:
        self._critic_optimizer = critic_optimizer
