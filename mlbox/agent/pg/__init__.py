from torch.nn import Module
from torch.optim import Optimizer

from mlbox.agent import Agent
from mlbox.types import T_Action, T_Obs


class PGAgent(Agent[T_Obs, T_Action]):
    def __init__(self) -> None:
        super().__init__()

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
    def optimizer(self) -> Optimizer:
        try:
            return self._optimizer
        except AttributeError:
            raise NotImplementedError('optimizer') from None

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer
