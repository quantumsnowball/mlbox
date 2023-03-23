import random
from collections import deque
from dataclasses import astuple, dataclass
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

T_State = TypeVar('T_State')
T_Action = TypeVar('T_Action')
T_Reward = TypeVar('T_Reward')


@dataclass
class Experience(Generic[T_State,
                         T_Action,
                         T_Reward]):
    state: T_State
    action: T_Action
    reward: T_Reward
    next_state: T_State
    done: bool

    def tuple(self) -> tuple[T_State,
                             T_Action,
                             T_Reward,
                             T_State,
                             bool]:
        return astuple(self)


@dataclass
class Batch:
    states: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    next_states: npt.NDArray
    dones: npt.NDArray


class Replay(Generic[T_State,
                     T_Action,
                     T_Reward]):
    def __init__(self,
                 maxlen: int = 10000):
        self._memory = deque[Experience[T_State,
                                        T_Action,
                                        T_Reward]](maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._memory)

    def remember(self,
                 exp: Experience[T_State,
                                 T_Action,
                                 T_Reward]) -> None:
        self._memory.append(exp)

    def sample(self,
               batch_size: int) -> Batch:
        # sampling
        size = min(len(self._memory), batch_size)
        samples = random.sample(self._memory, size)
        # pack
        states = np.array(tuple(
            s.state for s in samples)).reshape((size, -1))
        actions = np.array(tuple(
            s.action for s in samples)).reshape((size, -1))
        rewards = np.array(tuple(
            s.reward for s in samples)).reshape((size, -1))
        next_states = np.array(tuple(
            s.next_state for s in samples)).reshape((size, -1))
        dones = np.array(tuple(
            s.done for s in samples)).reshape((size, -1))
        return Batch(states, actions, rewards, next_states, dones)
