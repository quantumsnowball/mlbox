import numpy as np
import torch


class OrnsteinUhlenbeckNoise:
    def __init__(self,
                 action_space: int,
                 mu: float = 0.0,
                 theta: float = 0.05,
                 sigma: float = 0.02):
        self.action_space = action_space
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x = np.ones(action_space) * self.mu

    def __call__(self) -> np.ndarray:
        self.x += self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.action_space)
        return self.x.copy()
