import itertools
from pathlib import Path

import torch
from torch import tensor
from torch.nn import MSELoss
from typing_extensions import override

from mlbox.agent.basic import BasicAgent
from mlbox.agent.pg.memory import Buffer
from mlbox.agent.pg.props import PGProps
from mlbox.events import TerminatedError
from mlbox.types import T_Action, T_Obs


class PGAgent(BasicAgent[T_Obs, T_Action], PGProps):
    reward_to_go = False
    baseline = False

    def __init__(self) -> None:
        super().__init__()
        self.buffer = Buffer[T_Obs, T_Action]()

    #
    # training
    #

    @override
    def learn(self) -> None:
        self.policy_net.train()
        b = self.buffer.get_batch(device=self.device)
        obs = b.obs
        action = b.action
        reward = b.reward_to_go if self.reward_to_go else b.reward_traj
        # weight
        if self.baseline:
            value = self.baseline_net(obs).squeeze(1)
            advantages = reward - value
            weight = advantages
        else:
            weight = reward

        # learning policy
        policy = self.policy_net(obs)
        log_prob = policy.log_prob(action)
        loss = -(log_prob*weight).mean()
        # backward
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        # learning state value
        if self.baseline:
            baseline_loss = MSELoss()(self.baseline_net(obs).squeeze(1), reward)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

    n_eps = 100
    batch_size = 6000

    @override
    def train(self) -> None:
        self.policy_net.train()
        self.reset_rolling_reward()
        # episode loop
        for i_eps in range(1, self.n_eps):
            try:
                # only train on current policy experience
                self.buffer.clear()
                # trajectory loop
                for i_batch in itertools.count():
                    # reset to a new environment
                    obs, *_ = self.env.reset()
                    # step loop
                    for _ in range(self.max_step):
                        # act
                        action = self.decide(obs)
                        # step
                        try:
                            next_obs, reward, terminated, truncated, *_ = \
                                self.env.step(action)
                        except TerminatedError:
                            break
                        done = terminated or truncated
                        # cache experience
                        self.buffer.cache(obs, action, reward)
                        # pointing next
                        obs = next_obs
                        if done:
                            break
                    # flush cache trajectory to memory
                    self.buffer.flush()
                    # report progress
                    self.print_progress_bar(i_batch)
                    # enough traj
                    if len(self.buffer) >= self.batch_size:
                        break
                # learn from current batch
                self.learn()
                # report progress
                self.print_validation_result(i_eps)
                self.render_showcase(i_eps)
            except KeyboardInterrupt:
                print(f'\nManually stopped training loop')
                break

    #
    # acting
    #

    @override
    def decide(self, obs: T_Obs) -> T_Action:
        with torch.no_grad():
            obs_tensor = tensor(obs, device=self.device)
            policy = self.policy_net(obs_tensor)
            action = policy.sample().cpu().numpy()
            return action

    #
    # I/O
    #
    @override
    def load(self,
             path: Path | str) -> None:
        path = Path(path)
        self.policy_net.load_state_dict(torch.load(path))
        print(f'Loaded model: {path}')

    @override
    def save(self,
             path: Path | str) -> None:
        path = Path(path)
        torch.save(self.policy_net.state_dict(), path)
        print(f'Saved model: {path}')
