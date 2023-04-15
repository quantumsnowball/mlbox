from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from typing_extensions import override

from mlbox.agent.basic import BasicAgent
from mlbox.agent.ddpg.memory import CachedReplay
from mlbox.agent.ddpg.props import DDPGProps
from mlbox.events import TerminatedError
from mlbox.types import T_Action, T_Obs


class DDPGAgent(BasicAgent[T_Obs, T_Action],
                DDPGProps):
    # replay memory
    replay_size = 10000

    def __init__(self) -> None:
        super().__init__()
        self.replay = CachedReplay[T_Obs, T_Action](self.replay_size)

    polyak = 0.9

    def update_targets(self) -> None:
        for dest_net, src_net in ((self.actor_net_target, self.actor_net),
                                  (self.critic_net_target, self.critic_net)):
            for dest, src in zip(dest_net.parameters(), src_net.parameters()):
                dest.data.copy_(self.polyak * src.data + (1.0 - self.polyak) * dest.data)

    def sync_targets(self) -> None:
        self.actor_net_target.load_state_dict(self.actor_net.state_dict())
        self.critic_net_target.load_state_dict(self.critic_net.state_dict())

    n_epoch = 1
    batch_size = 512
    gamma = 0.99

    @override
    def learn(self) -> None:
        for _ in range(self.n_epoch):
            # prepare batch of experience
            batch = self.replay.sample(self.batch_size, device=self.device)
            obs, action, reward, next_obs, terminated = batch.tuple
            reward = batch.reward.unsqueeze(1)
            terminated = batch.terminated.unsqueeze(1)
            # set train mode
            self.actor_net.train()
            self.critic_net.train()
            # calc target
            q_target = self.critic_net_target(next_obs, self.actor_net_target(next_obs))
            critic_target = reward + self.gamma*q_target*(~terminated)
            # calc currents
            critic = self.critic_net(obs, action)
            # critic backward
            self.critic_optimizer.zero_grad()
            critic_loss = F.mse_loss(critic_target, critic)
            critic_loss.backward()
            self.critic_optimizer.step()
            # actor backward
            self.actor_optimizer.zero_grad()
            actor_loss = -self.critic_net(obs, self.actor_net(obs)).mean()
            actor_loss.backward()
            self.actor_optimizer.step()

    n_eps = 100
    update_target_every = 1

    @override
    def train(self) -> None:
        self.sync_targets()
        self.actor_net.train()
        self.reset_rolling_reward()
        for i_eps in range(1, self.n_eps+1):
            try:
                self.progress = min(max(i_eps/self.n_eps, 0), 1)
                # reset to a new environment
                obs, *_ = self.env.reset()
                # run the env
                for _ in range(self.max_step):
                    # act
                    action = self.explore(obs, self.progress)
                    # step
                    try:
                        next_obs, reward, terminated, truncated, *_ = \
                            self.env.step(action)
                    except TerminatedError:
                        break
                    done = terminated or truncated
                    # cache experience
                    self.replay.cache(obs, action, reward, next_obs, terminated)
                    # pointing next
                    obs = next_obs
                    if done:
                        break
                # post processing to cached experience before flush
                self.replay.assert_terminated_flag()
                # flush cache experience to memory
                self.replay.flush()
                # learn from experience replay
                self.learn()
                if i_eps % self.update_target_every == 0:
                    self.update_targets()
                # report progress
                self.print_progress_bar(i_eps)
                self.print_validation_result(i_eps)
                self.render_showcase(i_eps)
            except KeyboardInterrupt:
                print(f'\nManually stopped training loop')
                break

    min_noise = 0.02
    max_noise = 2.0

    def explore(self, obs: T_Obs, progress: float) -> T_Action:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device)
            action = self.actor_net(obs_tensor)
            action = action.cpu().numpy()
            std = self.min_noise + (1-progress)*(self.max_noise-self.min_noise)
            noise = np.random.normal(0, std, action.shape)
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
