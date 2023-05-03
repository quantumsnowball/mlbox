from itertools import chain

from torch import Tensor
from torch.distributions import Categorical, Normal
from torch.nn import (BatchNorm1d, Dropout, Identity, Linear, Module,
                      ReLU, Sequential, Softplus, Tanh)


class ActorCriticDiscrete(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 Activation: type[Module] = ReLU,
                 batch_norm: bool = True,
                 dropout: float | None = 0.0,
                 hidden_dim: int = 64,
                 base_n: int = 0,
                 actor_n: int = 0,
                 critic_n: int = 0):
        super().__init__()
        # base layer
        self.base = Sequential(
            Linear(in_dim, hidden_dim),
            BatchNorm1d(hidden_dim) if batch_norm else Identity(),
            Activation(),
            Dropout(dropout) if dropout is not None else Identity(),
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation(),
                Dropout(dropout) if dropout is not None else Identity(),
            ) for _ in range(base_n)))
        )
        # actor layer
        self.actor = Sequential(
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation(),
                Dropout(dropout) if dropout is not None else Identity(),
            ) for _ in range(actor_n))),
            Linear(hidden_dim, out_dim),
        )
        # policy
        self.dist = Categorical  # DON'T define this inside forward()
        # critic
        self.critic = Sequential(
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation(),
                Dropout(dropout) if dropout is not None else Identity(),
            ) for _ in range(critic_n))),
            Linear(hidden_dim, 1),
        )

    def forward(self, obs: Tensor) -> tuple[Categorical, Tensor]:
        base_out = self.base(obs)
        logits = self.actor(base_out)
        policy = self.dist(logits=logits)
        value: Tensor = self.critic(base_out)
        return policy, value


class ActorCriticContinuous(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 Activation: type[Module] = ReLU,
                 batch_norm: bool = True,
                 dropout: float | None = 0.0,
                 hidden_dim: int = 64,
                 base_n: int = 0,
                 mu_n: int = 0,
                 sigma_n: int = 0,
                 critic_n: int = 0,
                 mu_clip: bool = False,
                 mu_scale: float = 1.0) -> None:
        super().__init__()
        # base
        self.base = Sequential(
            Linear(in_dim, hidden_dim),
            BatchNorm1d(hidden_dim) if batch_norm else Identity(),
            Activation(),
            Dropout(dropout) if dropout is not None else Identity(),
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation(),
                Dropout(dropout) if dropout is not None else Identity(),
            ) for _ in range(base_n)))
        )
        # mu
        self.mu = Sequential(
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation(),
                Dropout(dropout) if dropout is not None else Identity(),
            ) for _ in range(mu_n))),
            Linear(hidden_dim, out_dim),
            Tanh() if mu_clip else Identity()
        )
        # sigma
        self.sigma = Sequential(
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation(),
                Dropout(dropout) if dropout is not None else Identity(),
            ) for _ in range(sigma_n))),
            Linear(hidden_dim, out_dim),
            Softplus(),
        )
        # policy
        self.dist = Normal
        # value
        self.critic = Sequential(
            *chain(*((
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim) if batch_norm else Identity(),
                Activation(),
                Dropout(dropout) if dropout is not None else Identity(),
            ) for _ in range(critic_n))),
            Linear(hidden_dim, 1),
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
