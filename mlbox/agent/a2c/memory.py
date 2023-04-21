from collections import deque
from dataclasses import astuple, dataclass
from typing import Any, Generic, SupportsFloat

import numpy as np
import torch
import torch as T
from numpy.typing import NDArray
from torch import Tensor, tensor

from mlbox.types import T_Action, T_Obs


@dataclass
class Experience(Generic[T_Obs,
                         T_Action]):
    obs: T_Obs
    action: T_Action
    reward: float
    next_obs: T_Obs
    terminated: bool


ExperiencesTuple = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, ]


@dataclass
class Experiences:
    obs: Tensor
    action: Tensor
    reward: Tensor
    next_obs: Tensor
    terminated: Tensor

    @property
    def tuple(self) -> ExperiencesTuple:
        exp_tuple: ExperiencesTuple = astuple(self)
        return exp_tuple


class Buffer(Generic[T_Obs, T_Action]):
    def __init__(self) -> None:
        self._cached = deque[Experience[T_Obs,
                                        T_Action]]()
        self._memory = deque[Experience[T_Obs,
                                        T_Action]]()

    def __len__(self) -> int:
        return len(self._memory)

    def cache(self,
              obs: T_Obs,
              action: T_Action,
              reward: SupportsFloat,
              next_obs: T_Obs,
              terminated: bool) -> None:
        self._cached.append(
            Experience[T_Obs, T_Action](
                obs=obs,
                action=action,
                reward=float(reward),
                next_obs=next_obs,
                terminated=terminated,
            )
        )

    def flush(self) -> None:
        # ensure terminated flag
        last = self._cached[-1]
        last.terminated = True
        # flush
        self._memory.extend(self._cached)
        # reset cache
        self._cached.clear()

    def recall(self,
               device: T.device = T.device('cpu')) -> Experiences:
        def to_tensor(arr: NDArray[Any],
                      **kwargs: Any) -> Tensor:
            return tensor(arr, device=device, **kwargs)
        return Experiences(
            obs=to_tensor(np.stack([s.obs
                                    for s in self._memory])),
            action=to_tensor(np.array([s.action
                                       for s in self._memory])),
            reward=to_tensor(np.array([s.reward
                                       for s in self._memory]), dtype=torch.float32),
            next_obs=to_tensor(np.stack([s.next_obs
                                         for s in self._memory])),
            terminated=to_tensor(np.stack([s.terminated
                                           for s in self._memory])),
        )

    def clear(self) -> None:
        self._memory.clear()
        self._cached.clear()
