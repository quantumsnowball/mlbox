from pathlib import Path

import numpy as np
import torch
from typing_extensions import override

from mlbox.agent.basic import BasicAgent
from mlbox.agent.ddpg.props import DDPGProps
from mlbox.types import T_Action, T_Obs


class DDPGAgent(BasicAgent[T_Obs, T_Action],
                DDPGProps):
    def __init__(self) -> None:
        super().__init__()

    polyak = 0.9

    def update_targets(self) -> None:
        for target_net, src_net in ((self.actor_net_target, self.actor_net),
                                    (self.critic_net_target, self.critic_net)):
            for target, src in zip(target_net.parameters(), src_net.parameters()):
                target.data.copy_(self.polyak * src.data + (1.0 - self.polyak) * target.data)

    @override
    def learn(self) -> None:
        pass

    @override
    def train(self) -> None:
        pass

    def explore(self, obs: T_Obs) -> T_Action:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device)
            action = self.actor_net(obs_tensor)
            action = action.cpu().numpy()
            sd = 0.02  # TODO
            noise = np.random.normal(0, sd, action.shape)
            return action + noise

    @override
    def decide(self, obs: T_Obs) -> T_Action:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device)
            action = self.actor_net(obs_tensor)
            return action.cpu().numpy()

    #
    # I/O
    #

    @override
    def load(self,
             path: Path | str) -> None:
        path = Path(path)
        state = torch.load(path)
        self.actor_net.load_state_dict(state['actor'])
        self.critic_net.load_state_dict(state['critic'])
        print(f'Loaded model: {path}')

    @override
    def save(self,
             path: Path | str) -> None:
        path = Path(path)
        state = dict(actor=self.actor_net.state_dict(),
                     critic=self.critic_net.state_dict())
        torch.save(state, path)
        print(f'Saved model: {path}')
