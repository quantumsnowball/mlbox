from torch.nn import Module
from torch.optim import Optimizer

from mlbox.utils.wrapper import assured


class PGProps:
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
