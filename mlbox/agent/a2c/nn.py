import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal


class ActorCriticDiscrete(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 hidden_dim: int = 64,
                 base_n: int = 0,
                 actor_n: int = 0,
                 critic_n: int = 0):
        super().__init__()
        # base layer
        self.base = nn.Sequential()
        self.base.append(nn.Linear(in_dim, hidden_dim))
        self.base.append(nn.ReLU())
        for _ in range(base_n):
            self.base.append(nn.Linear(hidden_dim, hidden_dim))
            self.base.append(nn.ReLU())
        # actor layer
        self.actor = nn.Sequential()
        for _ in range(actor_n):
            self.actor.append(nn.Linear(hidden_dim, hidden_dim))
            self.actor.append(nn.ReLU())
        self.actor.append(nn.Linear(hidden_dim, out_dim))
        # policy
        self.dist = Categorical  # DON'T define this inside forward()
        # critic
        self.critic = nn.Sequential()
        for _ in range(critic_n):
            self.critic.append(nn.Linear(hidden_dim, hidden_dim))
            self.critic.append(nn.ReLU())
        self.critic.append(nn.Linear(hidden_dim, 1))

    def forward(self, obs: Tensor):
        base_out = self.base(obs)
        logits = self.actor(base_out)
        policy = self.dist(logits=logits)
        value = self.critic(base_out)
        return policy, value


class ActorCriticContinuous(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 hidden_dim: int = 64,
                 base_n: int = 0,
                 mu_n: int = 0,
                 sigma_n: int = 0,
                 critic_n: int = 0,
                 mu_clip: bool = False,
                 mu_scale: float = 1.0) -> None:
        super().__init__()
        # base
        self.base = nn.Sequential()
        self.base.append(nn.Linear(in_dim, hidden_dim))
        self.base.append(nn.ReLU())
        for _ in range(base_n):
            self.base.append(nn.Linear(hidden_dim, hidden_dim))
            self.base.append(nn.ReLU())
        # mu
        self.mu = nn.Sequential()
        for _ in range(mu_n):
            self.mu.append(nn.Linear(hidden_dim, hidden_dim))
            self.mu.append(nn.ReLU())
        self.mu.append(nn.Linear(hidden_dim, out_dim))
        if mu_clip:
            self.mu.append(nn.Tanh())
        # sigma
        self.sigma = nn.Sequential()
        for _ in range(sigma_n):
            self.sigma.append(nn.Linear(hidden_dim, hidden_dim))
            self.sigma.append(nn.ReLU())
        self.sigma.append(nn.Linear(hidden_dim, out_dim))
        self.sigma.append(nn.Softplus())
        # policy
        self.dist = Normal
        # value
        self.critic = nn.Sequential()
        for _ in range(critic_n):
            self.critic.append(nn.Linear(hidden_dim, hidden_dim))
            self.critic.append(nn.ReLU())
        self.critic.append(nn.Linear(hidden_dim, 1))
        # const
        self.mu_scale = mu_scale

    def forward(self, obs: Tensor):
        base_out = self.base(obs)
        mu = self.mu(base_out) * self.mu_scale
        sigma = self.sigma(base_out)
        policy = self.dist(mu, sigma)
        value = self.critic(base_out)
        return policy, value
