from pathlib import Path

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from typing_extensions import override

from mlbox.agent import BasicAgent
from mlbox.agent.a2c.memory import Buffer
from mlbox.agent.a2c.props import Props
from mlbox.events import TerminatedError
from mlbox.types import T_Action, T_Obs


class A2CAgent(Props[T_Obs, T_Action],
               BasicAgent[T_Obs, T_Action]):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = Buffer[T_Obs, T_Action]()

    #
    # core
    #

    gamma = 0.99
    lr = 1e-3
    value_weight = 0.5

    @override
    def learn(self) -> None:
        # set mode
        self.use_train_mode()
        # batch
        obs, action, reward, next_obs, terminated = self.buffer.recall(self.device).tuple
        # actor and critic
        policy, value = self.actor_critic_net(obs)
        _, next_value = self.actor_critic_net(next_obs)
        value_target = reward.unsqueeze(1) + self.gamma * next_value * (~terminated.unsqueeze(1))
        delta = value_target - value
        # Compute the policy loss and value loss
        log_prob = policy.log_prob(action).unsqueeze(1)
        advantage = delta.detach()
        policy_loss = -(log_prob * advantage).mean()
        value_loss = delta.pow(2).mean()
        loss = policy_loss + self.value_weight*value_loss
        # Update the actor-critic network using the combined loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.actor_critic_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        # reset mode
        self.use_eval_mode()

    n_eps = 1000

    @override
    def train(self) -> None:
        self.reset_rolling_reward()
        self.reset_eps_timer()
        # episode loop
        for i_eps in range(1, self.n_eps+1):
            try:
                # only train on current policy experience
                self.buffer.clear()
                # reset to a new environment
                obs, *_ = self.env.reset()
                obs = self.encode_obs(obs)
                # step loop
                for _ in range(self.max_step):
                    # act
                    action = self.decide(obs)
                    # step
                    try:
                        next_obs, reward, terminated, truncated, *_ = self.env.step(action)
                        next_obs = self.encode_obs(next_obs)
                    except TerminatedError:
                        break
                    done = terminated or truncated
                    # cache experience
                    self.buffer.cache(obs, action, reward, next_obs, done)
                    # pointing next
                    obs = next_obs
                    if done:
                        break
                # flush cache trajectory to memory
                self.buffer.flush()
                # learn from current batch
                self.learn()
                # report progress
                self.print_progress_bar(i_eps)
                self.print_evaluation_result(i_eps)
                self.render_showcase(i_eps)
            except KeyboardInterrupt:
                print(f'\nManually stopped training loop')
                break

    #
    # acting
    #

    @override
    def decide(self, obs: T_Obs) -> T_Action:
        self.use_eval_mode()
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device).unsqueeze(0)
            policy, _ = self.actor_critic_net(obs_tensor)
            action: T_Action = policy.sample().squeeze(0).cpu().numpy()
            return action

    #
    # I/O
    #

    @override
    def load(self,
             path: Path | str) -> None:
        path = Path(path)
        self.actor_critic_net.load_state_dict(torch.load(path))

    @override
    def save(self,
             path: Path | str) -> None:
        path = Path(path)
        torch.save(self.actor_critic_net.state_dict(), path)
