from itertools import chain

import torch.nn as nn
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
        self.base = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            *chain(*((
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(base_n)))
        )
        # actor layer
        self.actor = nn.Sequential(
            *chain(*((
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(actor_n))),
            nn.Linear(hidden_dim, out_dim),
        )
        # policy
        self.dist = Categorical  # DON'T define this inside forward()
        # critic
        self.critic = nn.Sequential(
            *chain(*((
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(critic_n))),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: Tensor) -> tuple[Categorical, Tensor]:
        base_out = self.base(obs)
        logits = self.actor(base_out)
        policy = self.dist(logits=logits)
        value: Tensor = self.critic(base_out)
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
        self.base = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            *chain(*((
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(base_n)))
        )
        # mu
        self.mu = nn.Sequential(
            *chain(*((
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(mu_n))),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh() if mu_clip else nn.Identity()
        )
        # sigma
        self.sigma = nn.Sequential(
            *chain(*((
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(sigma_n))),
            nn.Linear(hidden_dim, out_dim),
            nn.Softplus(),
        )
        # policy
        self.dist = Normal
        # value
        self.critic = nn.Sequential(
            *chain(*((
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(critic_n))),
            nn.Linear(hidden_dim, 1),
        )
        # const
        self.mu_scale = mu_scale

    def forward(self, obs: Tensor) -> tuple[Normal, Tensor]:
        base_out = self.base(obs)
        mu = self.mu(base_out) * self.mu_scale
        sigma = self.sigma(base_out)
        policy = self.dist(mu, sigma)
        value: Tensor = self.critic(base_out)
        return policy, value
