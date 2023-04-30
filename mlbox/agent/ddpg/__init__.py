from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, tensor
from torch.nn import Module
from typing_extensions import override

from mlbox.agent import BasicAgent
from mlbox.agent.ddpg.memoryondevice import CachedReplay
from mlbox.agent.ddpg.props import Props
from mlbox.events import TerminatedError
from mlbox.types import T_Action, T_Obs


class DDPGAgent(Props[T_Obs, T_Action],
                BasicAgent[T_Obs, T_Action]):
    # replay memory
    replay_size = 10000

    def __init__(self) -> None:
        super().__init__()
        self.replay = CachedReplay[T_Obs, T_Action](self.replay_size,
                                                    device=self.device)
        self.min_action = 0
        self.max_action = 1

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
        # set mode
        self.use_train_mode()
        for i_epoch, batch in enumerate(self.replay.dataloader(self.batch_size)):
            # prepare batch of experience
            obs, action, reward, next_obs, terminated = batch.tuple
            reward = batch.reward.unsqueeze(1)
            terminated = batch.terminated.unsqueeze(1)
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
            #
            if self.n_epoch > 0 and i_epoch + 1 >= self.n_epoch:
                break

        # reset mode
        self.use_eval_mode()

    n_eps = 100
    update_target_every = 1

    @override
    def train(self) -> None:
        self.log_graphs()
        self.sync_targets()
        self.reset_rolling_reward()
        self.reset_eps_timer()
        for i_eps in range(1, self.n_eps+1):
            try:
                # progress
                self.progress = min(max(i_eps/self.n_eps, 0), 1)
                # reset to a new environment
                obs, *_ = self.env.reset()
                # run the env
                for _ in range(self.max_step):
                    # act
                    action = self.explore(obs, self.progress)
                    # step
                    try:
                        next_obs, reward, terminated, truncated, *_ = self.env.step(action)
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
                self.print_evaluation_result(i_eps)
                self.render_showcase(i_eps)
                # tensorboard
                self.log_histogram(i_eps)
            except KeyboardInterrupt:
                print(f'\nManually stopped training loop')
                break
        self.close_writer()

    min_noise = 0.02
    max_noise = 2.0

    def explore(self, obs: T_Obs, progress: float) -> T_Action:
        self.use_eval_mode()
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device)
            action = self.actor_net(obs_tensor.unsqueeze(0)).squeeze(0)
            action = action.cpu().numpy()
            std = self.min_noise + (1-progress)*(self.max_noise-self.min_noise)
            noise = np.random.normal(0, std, action.shape)
            action_with_noise: T_Action = np.clip(action + noise, self.min_action, self.max_action)
            return action_with_noise

    @override
    def decide(self, obs: T_Obs) -> T_Action:
        self.use_eval_mode()
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device)
            action_tensor: Tensor = self.actor_net(obs_tensor.unsqueeze(0)).squeeze(0)
            action: T_Action = action_tensor.cpu().numpy()
            return action

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

    @override
    def save(self,
             path: Path | str) -> None:
        path = Path(path)
        state = dict(actor=self.actor_net.state_dict(),
                     critic=self.critic_net.state_dict())
        torch.save(state, path)

    #
    # tensorboard
    #
    def close_writer(self) -> None:
        if not self.tensorboard:
            return
        self.writer.close()

    def log_graphs(self) -> None:
        if not self.tensorboard:
            return

        class DisplayNet(Module):
            def __init__(self, actor_net: Module, critic_net: Module) -> None:
                super().__init__()
                self.actor_net = actor_net
                self.critic_net = critic_net

            def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
                action = self.actor_net(obs)
                value = self.critic_net(obs, action)
                return action, value
        obs_sample = tensor(self.env.observation_space.sample(), device=self.device)
        self.writer.add_graph(DisplayNet(self.actor_net, self.critic_net), obs_sample)

    log_histogram_every = 10

    def log_histogram(self, i_eps: int) -> None:
        if not self.tensorboard:
            return
        if i_eps % self.log_histogram_every == 0:
            for net in (self.actor_net, self.critic_net):
                for name, param in net.named_parameters():
                    self.writer.add_histogram(name, param, global_step=i_eps)
