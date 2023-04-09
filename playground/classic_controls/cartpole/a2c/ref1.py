import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


# Define the actor and critic networks using PyTorch
class ActorCritic(nn.Module):
    def __init__(self, input_size, num_actions):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


# Initialize the environment and set the hyperparameters
env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
num_actions = env.action_space.n
lr = 0.001
gamma = 0.99

# Initialize the actor-critic network and optimizer
net = ActorCritic(input_size, num_actions)
optimizer = optim.Adam(net.parameters(), lr=lr)

# Define the A2C update function


def a2c_update(states, actions, rewards, next_states, done):
    # Convert the inputs to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.float32)

    # Compute the predicted policy and value for the current state
    policy, value = net(states)

    # Compute the advantage values using the critic network
    next_value = net(next_states)[1]
    delta = rewards + gamma * next_value * (1 - done) - value
    advantage = delta.detach()

    # Compute the policy loss and value loss
    log_prob = torch.log(policy.gather(1, actions.unsqueeze(1)))
    policy_loss = -(log_prob * advantage).mean()
    value_loss = delta.pow(2).mean()

    # Update the actor-critic network using the combined loss
    loss = policy_loss + 0.5 * value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Train the actor-critic network using the A2C update function
for episode in range(1000):
    state, *_ = env.reset()
    done = False
    states, actions, rewards, next_states = [], [], [], []

    while not done:
        # Select an action using the policy network
        policy, _ = net(torch.tensor(state, dtype=torch.float32))
        action = torch.multinomial(policy, 1).item()

        # Take a step in the environment and record the transition
        next_state, reward, done, *_ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        state = next_state

    # Update the actor-critic network using the A2C update function
    a2c_update(states, actions, rewards, next_states, done)
