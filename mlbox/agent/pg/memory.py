from collections import deque
from dataclasses import dataclass
from typing import Any, Generic, SupportsFloat

import numpy as np
from numpy.typing import NDArray
from torch import Tensor, tensor

from mlbox.types import T_Action, T_Obs


@dataclass
class Experience(Generic[T_Obs,
                         T_Action]):
    obs: T_Obs
    action: T_Action
    reward: SupportsFloat
    traj_reward: float = 0

    def __post_init__(self) -> None:
        # post processing
        self.reward = np.float32(self.reward)


@dataclass
class Batch:
    obs: Tensor
    action: Tensor
    traj_reward: Tensor


class Buffer(Generic[T_Obs, T_Action]):
    def __init__(self):
        self._cached = deque[Experience[T_Obs,
                                        T_Action]]()
        self._memory = deque[Experience[T_Obs,
                                        T_Action]]()

    def __len__(self) -> int:
        return len(self._memory)

    def cache(self,
              obs: T_Obs,
              action: T_Action,
              reward: SupportsFloat) -> None:
        self._cached.append(
            Experience[T_Obs, T_Action](
                obs=obs,
                action=action,
                reward=reward,
            )
        )

    def flush(self) -> None:
        traj_reward = sum([float(s.reward) for s in self._cached])
        # fill traj_reward
        for sample in self._cached:
            sample.traj_reward = traj_reward
        # flush
        self._memory.extend(self._cached)
        # reset cache
        self._cached.clear()

    def get_batch(self,
                  device: str = 'cpu') -> Batch:
        def to_tensor(arr: NDArray[Any]) -> Tensor:
            return tensor(arr, device=device)
        return Batch(
            obs=to_tensor(np.stack([s.obs for s in self._memory])),
            action=to_tensor(np.array([s.action for s in self._memory])),
            traj_reward=to_tensor(np.array([s.traj_reward
                                            for s in self._memory])),
        )

    def clear(self) -> None:
        self._memory.clear()
        self._cached.clear()
