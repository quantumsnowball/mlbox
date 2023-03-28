import random
from collections import deque
from dataclasses import dataclass
from typing import Generic, SupportsFloat, TypeVar

import numpy as np
import numpy.typing as npt

T_Obs = TypeVar('T_Obs')
T_Action = TypeVar('T_Action')


@dataclass
class Experience(Generic[T_Obs,
                         T_Action]):
    obs: T_Obs
    action: T_Action
    reward: SupportsFloat
    next_obs: T_Obs
    terminated: bool

    def tuple(self) -> tuple[T_Obs,
                             T_Action,
                             SupportsFloat,
                             T_Obs,
                             bool]:
        return (self.obs, self.action, self.reward, self.next_obs, self.terminated, )


@dataclass
class Batch:
    obs: npt.NDArray[np.float32]
    action: npt.NDArray[np.float32]
    reward: npt.NDArray[np.float32]
    next_obs: npt.NDArray[np.float32]
    done: npt.NDArray[np.bool8]


class Replay(Generic[T_Obs,
                     T_Action]):
    def __init__(self,
                 maxlen: int = 10000):
        self._memory = deque[Experience[T_Obs,
                                        T_Action]](maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._memory)

    def remember(self,
                 exp: Experience[T_Obs,
                                 T_Action]) -> None:
        self._memory.append(exp)

    def sample(self,
               batch_size: int) -> Batch:
        # sampling
        size = min(len(self._memory), batch_size)
        samples = random.sample(self._memory, size)
        # pack
        obs = np.array(tuple(
            s.obs for s in samples)).reshape((size, -1))
        action = np.array(tuple(
            s.action for s in samples)).reshape((size, -1))
        reward = np.array(tuple(
            s.reward for s in samples)).reshape((size, -1))
        next_obs = np.array(tuple(
            s.next_obs for s in samples)).reshape((size, -1))
        done = np.array(tuple(
            s.terminated for s in samples)).reshape((size, -1))
        return Batch(obs, action, reward, next_obs, done)
