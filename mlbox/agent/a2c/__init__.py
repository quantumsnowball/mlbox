from collections import deque
from pathlib import Path

import torch
from torch.distributions import Categorical
from typing_extensions import override

from mlbox.agent.a2c.memory import Buffer
from mlbox.agent.a2c.noise import OrnsteinUhlenbeckNoise
from mlbox.agent.a2c.props import A2CProps
from mlbox.agent.basic import BasicAgent
from mlbox.events import TerminatedError
from mlbox.types import T_Action, T_Obs


class A2CAgent(BasicAgent[T_Obs, T_Action],
               A2CProps):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = Buffer[T_Obs, T_Action]()

    #
    # I/O
    #

    @override
    def load(self,
             path: Path | str) -> None:
        path = Path(path)
        self.actor_critic_net.load_state_dict(torch.load(path))
        print(f'Loaded model: {path}')

    @override
    def save(self,
             path: Path | str) -> None:
        path = Path(path)
        torch.save(self.actor_critic_net.state_dict(), path)
        print(f'Saved model: {path}')


class A2CDiscreteAgent(A2CAgent[T_Obs, T_Action]):
    gamma = 0.99
    lr = 1e-3

    @override
    def learn(self) -> None:
        self.actor_critic_net.train()
        # batch
        obs, action, reward, next_obs, terminated = self.buffer.recall(self.device).tuple
        # actor and critic
        policy, value = self.actor_critic_net(obs)
        _, next_value = self.actor_critic_net(next_obs)
        value_target = reward.unsqueeze(1) + self.gamma * next_value * (~terminated.unsqueeze(1))
        delta = value_target - value
        # Compute the policy loss and value loss
        log_prob = torch.log(policy.gather(1, action.unsqueeze(1)))
        advantage = delta.detach()
        policy_loss = -(log_prob * advantage).mean()
        value_loss = delta.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss
        # Update the actor-critic network using the combined loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    n_eps = 1000
    max_step = 500
    print_hash_every = 10
    rolling_reward_ma = 5
    report_progress_every = 10

    @override
    def train(self) -> None:
        self.actor_critic_net.train()
        # episode loop
        rolling_reward = deque[float](maxlen=self.rolling_reward_ma)
        for i_eps in range(1, self.n_eps+1):
            try:
                # only train on current policy experience
                self.buffer.clear()
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
                if i_eps % self.print_hash_every == 0:
                    print('#', end='', flush=True)
                # evulate and report progress
                if i_eps % self.report_progress_every == 0:
                    rolling_reward.append(self.play(self.max_step))
                    mean_reward = sum(rolling_reward)/len(rolling_reward)
                    print(f' | Episode {i_eps:>4d} | {mean_reward=:.1f}')
            except KeyboardInterrupt:
                print(f'\nManually stopped training loop')
                break

    #
    # acting
    #

    @override
    def decide(self, obs: T_Obs) -> T_Action:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device)
            action_probs, _ = self.actor_critic_net(obs_tensor)
            dist = Categorical(action_probs)
            result = dist.sample().cpu().numpy()
            return result


class A2CContinuousAgent(A2CAgent[T_Obs, T_Action]):
    def __init__(self) -> None:
        super().__init__()
        self.ou_noise = OrnsteinUhlenbeckNoise(2)

    gamma = 0.99
    lr = 1e-3

    @override
    def learn(self) -> None:
        self.actor_critic_net.train()
        # batch
        obs, action, reward, next_obs, terminated = self.buffer.recall(self.device).tuple
        # actor and critic
        policy, value = self.actor_critic_net(obs)
        _, next_value = self.actor_critic_net(next_obs)
        value_target = reward.unsqueeze(1) + self.gamma * next_value * (~terminated.unsqueeze(1))
        delta = value_target - value
        # Compute the policy loss and value loss
        log_prob = policy.log_prob(action)
        advantage = delta.detach()
        policy_loss = -(log_prob * advantage).mean()
        value_loss = delta.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss
        # Update the actor-critic network using the combined loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    n_eps = 1000
    max_step = 500
    print_hash_every = 10
    rolling_reward_ma = 5
    report_progress_every = 10

    @override
    def train(self) -> None:
        self.actor_critic_net.train()
        # episode loop
        rolling_reward = deque[float](maxlen=self.rolling_reward_ma)
        for i_eps in range(1, self.n_eps+1):
            try:
                # only train on current policy experience
                self.buffer.clear()
                # reset to a new environment
                obs, *_ = self.env.reset()
                # step loop
                for _ in range(self.max_step):
                    # act
                    action = self.decide(obs)
                    action += self.ou_noise()
                    # step
                    try:
                        next_obs, reward, terminated, truncated, *_ = self.env.step(action)
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
                if i_eps % self.print_hash_every == 0:
                    print('#', end='', flush=True)
                # evulate and report progress
                if i_eps % self.report_progress_every == 0:
                    rolling_reward.append(self.play(self.max_step))
                    mean_reward = sum(rolling_reward)/len(rolling_reward)
                    print(f' | Episode {i_eps:>4d} | {mean_reward=:.1f}')
            except KeyboardInterrupt:
                print(f'\nManually stopped training loop')
                break

    #
    # acting
    #

    @override
    def decide(self, obs: T_Obs) -> T_Action:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device)
            policy, _ = self.actor_critic_net(obs_tensor)
            action = policy.sample().cpu().numpy()
            action += self.ou_noise()
            return action
