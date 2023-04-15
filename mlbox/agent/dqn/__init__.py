from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from typing_extensions import override

from mlbox.agent import EpsilonGreedyStrategy
from mlbox.agent.basic import BasicAgent
from mlbox.agent.dqn.memory import CachedReplay
from mlbox.agent.dqn.props import DQNProps
from mlbox.events import TerminatedError
from mlbox.types import T_Action, T_Obs


class DQNAgent(BasicAgent[T_Obs, T_Action],
               EpsilonGreedyStrategy[T_Obs, T_Action],
               DQNProps):
    # replay memory
    replay_size = 10000

    def __init__(self) -> None:
        super().__init__()
        self.replay = CachedReplay[T_Obs, T_Action](self.replay_size)

    #
    # training
    #
    def update_target(self) -> None:
        weights = self.policy.state_dict()
        self.target.load_state_dict(weights)

    n_epoch = 1000
    batch_size = 512
    gamma = 0.99

    @override
    def learn(self) -> None:
        for _ in range(self.n_epoch):
            # prepare batch of experience
            batch = self.replay.sample(self.batch_size, device=self.device)
            obs = batch.obs
            action = batch.action.unsqueeze(1)
            reward = batch.reward.unsqueeze(1)
            next_obs = batch.next_obs
            non_final_mask = (~batch.terminated)
            non_final_next_obs = next_obs[non_final_mask]
            # set train mode
            self.policy.train()
            # calc state-action value
            sa_val = self.policy(obs).gather(1, action)
            # calc expected state-action value
            next_sa_val = torch.zeros(len(obs), 1, device=self.device)
            with torch.no_grad():
                next_sa_val[non_final_mask] = self.target(
                    non_final_next_obs).max(1).values.unsqueeze(1)
            expected_sa_val = reward + self.gamma*next_sa_val
            # calc loss
            loss = self.loss_function(sa_val,
                                      expected_sa_val)
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

    n_eps = 100
    update_target_every = 10

    @ override
    def train(self) -> None:
        self.policy.train()
        self.reset_rolling_reward()
        for i_eps in range(1, self.n_eps+1):
            try:
                self.progress = min(max(i_eps/self.n_eps, 0), 1)
                # reset to a new environment
                obs, *_ = self.env.reset()
                # run the env
                for _ in range(self.max_step):
                    # act
                    action = self.decide(obs, epsilon=self.progress)
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
                    self.update_target()
                # report progress
                self.print_progress_bar(i_eps)
                self.print_validation_result(i_eps)
                self.render_showcase(i_eps)
            except KeyboardInterrupt:
                print(f'\nManually stopped training loop')
                break

    #
    # acting
    #

    @override
    def explore(self) -> T_Action:
        random_action = self.env.action_space.sample()
        return random_action

    @override
    def exploit(self, obs: T_Obs) -> T_Action:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device)
            best_value_action = torch.argmax(self.policy(obs_tensor))
            result = best_value_action.cpu().numpy()
            return result

    @override
    def decide(self,
               obs: T_Obs,
               *,
               epsilon: float = 1.0) -> T_Action:
        if np.random.random() > epsilon:
            return self.explore()
        else:
            return self.exploit(obs)

    #
    # I/O
    #

    @override
    def load(self,
             path: Path | str) -> None:
        path = Path(path)
        self.policy.load_state_dict(torch.load(path))
        self.update_target()
        print(f'Loaded model: {path}')

    @override
    def save(self,
             path: Path | str) -> None:
        path = Path(path)
        torch.save(self.policy.state_dict(), path)
        print(f'Saved model: {path}')
