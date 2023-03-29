from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt
from gymnasium import Env, Space
from gymnasium.spaces import Box, Discrete

Obs = npt.NDArray[np.float32]
Action = np.int64
Reward = np.float32

T_Obs = TypeVar('T_Obs')
T_Action = TypeVar('T_Action')


# lib
T_ActionSpace = TypeVar('T_ActionSpace', bound=Space)


class TrEnv(Env[T_Obs, T_Action]):
    ...


class Agent(Generic[T_Obs, T_Action]):
    def __init__(self) -> None:
        super().__init__()
        self.env: Env[T_Obs, T_Action]


class DQNAgent(Agent[T_Obs, T_Action]):
    def __init__(self) -> None:
        super().__init__()

    def explore(self) -> T_Action:
        return self.env.action_space.sample()

# implement


class MyEnv(TrEnv[Obs, Action]):
    action_space: Discrete = Discrete(3)


class MyAgent(DQNAgent[Obs, Action, ]):
    def __init__(self) -> None:
        super().__init__()
        self.env = MyEnv()
        n = self.env.action_space.n
        print(n)

    def main(self) -> None:
        self.env


agent = MyAgent()
agent.env.action_space.sample()
print(type(agent.env))
