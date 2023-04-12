import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal


class ActorCriticDiscrete(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        # actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )
        # policy
        self.dist = Categorical  # DON'T define this inside forward()
        # critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: Tensor):
        base_out = self.base(obs)
        logits = self.actor(base_out)
        policy = self.dist(logits=logits)
        value = self.critic(base_out)
        return policy, value


class ActorCriticContinuous(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64) -> None:
        super().__init__()
        # base
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # mu, sigma
        self.mu = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        self.sigma = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),
        )
        # value
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: Tensor):
        base_out = self.base(obs)
        mu = self.mu(base_out) * 2
        sigma = self.sigma(base_out)
        policy = Normal(mu, sigma)
        value = self.critic(base_out)
        return policy, value
