
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from mlbox.utils.wrapper import assured


class DQNProps:
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
