import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ActorCriticDiscrete(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.actor_fc1 = nn.Linear(input_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, output_dim)
        self.critic_fc1 = nn.Linear(input_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs: Tensor):
        # action probs
        actor_x = F.relu(self.actor_fc1(obs))
        action_probs = F.softmax(self.actor_fc2(actor_x), dim=-1)
        # state value
        critic_x = F.relu(self.critic_fc1(obs))
        state_value = self.critic_fc2(critic_x)
        # return
        return action_probs, state_value
