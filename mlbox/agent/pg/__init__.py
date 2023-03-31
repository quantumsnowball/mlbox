from pathlib import Path

import torch
from torch.distributions import Categorical
from torch.nn import Module
from typing_extensions import override

from mlbox.agent.basic import BasicAgent
from mlbox.types import T_Action, T_Obs


class PGAgent(BasicAgent[T_Obs, T_Action]):
    def __init__(self) -> None:
        super().__init__()

    #
    # props
    #

    @property
    def policy_net(self) -> Module:
        try:
            return self._policy_net
        except AttributeError:
            raise NotImplementedError('policy') from None

    @policy_net.setter
    def policy_net(self, policy_net: Module) -> None:
        self._policy_net = policy_net

    #
    # training
    #

    def policy(self,
               obs: T_Obs) -> Categorical:
        logits = self.policy_net(obs)
        return Categorical(logits=logits)

    @override
    def learn(self) -> None:
        raise NotImplementedError()

    @override
    def train(self) -> None:
        raise NotImplementedError()

    #
    # acting
    #

    @override
    def exploit(self, obs: T_Obs) -> T_Action:
        with torch.no_grad():
            result: T_Action = self.policy(obs).sample().item()
            return result

    #
    # I/O
    #

    @override
    def load(self,
             path: Path | str) -> None:
        raise NotImplementedError()

    @override
    def save(self,
             path: Path | str) -> None:
        raise NotImplementedError()
