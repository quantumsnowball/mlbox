from collections import deque
from collections.abc import Callable
from dataclasses import astuple, dataclass
from typing import Any, Generic, SupportsFloat

import numpy as np
import torch as T
from numpy.typing import NDArray
from torch import Tensor, tensor
from torch.utils.data import DataLoader, Dataset

from mlbox.types import T_Action, T_Obs


@dataclass
class Experience:
    obs: Tensor
    action: Tensor
    reward: Tensor
    next_obs: Tensor
    terminated: Tensor


@dataclass
class Batch:
    obs: Tensor
    action: Tensor
    reward: Tensor
    next_obs: Tensor
    terminated: Tensor

    @property
    def tuple(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, ]:
        return (self.obs, self.action, self.reward, self.next_obs, self.terminated)


class Replay(Dataset[Experience],
             Generic[T_Obs, T_Action]):
    def __init__(self,
                 maxlen: int = 10000,
                 *,
                 device: T.device = T.device('cpu')):
        self.device = device
        self._memory = deque[Experience](maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._memory)

    def __getitem__(self,
                    index: int) -> Experience:
        return self._memory[index]

    def remember(self,
                 obs: T_Obs,
                 action: T_Action,
                 reward: SupportsFloat,
                 next_obs: T_Obs,
                 terminated: bool) -> None:
        self._memory.append(
            Experience(
                obs=tensor(obs, device=self.device, dtype=T.float32),
                action=tensor(action, device=self.device, dtype=T.float32),
                reward=tensor(reward, device=self.device, dtype=T.float32),
                next_obs=tensor(next_obs, device=self.device, dtype=T.float32),
                terminated=tensor(terminated, device=self.device, dtype=T.bool),
            )
        )

    def dataloader(self,
                   batch_size: int) -> DataLoader[Experience]:
        def collate(batch: list[Experience]) -> Batch:
            return Batch(
                obs=T.stack([b.obs for b in batch]),
                action=T.stack([b.action for b in batch]),
                reward=T.stack([b.reward for b in batch]),
                next_obs=T.stack([b.next_obs for b in batch]),
                terminated=T.stack([b.terminated for b in batch])
            )
        loader = DataLoader[Experience](self,
                                        batch_size,
                                        shuffle=True,
                                        collate_fn=collate)
        return loader


class CachedReplay(Replay[T_Obs, T_Action]):
    def __init__(self,
                 maxlen: int = 10000,
                 *,
                 device: T.device = T.device('cpu')):
        super().__init__(maxlen, device=device)
        self._cached = deque[Experience]()

    def cache(self,
              obs: T_Obs,
              action: T_Action,
              reward: SupportsFloat,
              next_obs: T_Obs,
              terminated: bool) -> None:
        self._cached.append(
            Experience(
                obs=tensor(obs, device=self.device, dtype=T.float32),
                action=tensor(action, device=self.device, dtype=T.float32),
                reward=tensor(reward, device=self.device, dtype=T.float32),
                next_obs=tensor(next_obs, device=self.device, dtype=T.float32),
                terminated=tensor(terminated, device=self.device, dtype=T.bool),
            )
        )

    def flush(self) -> None:
        self._memory.extend(self._cached)
        self._cached.clear()

    # post processing helpers
    def assert_terminated_flag(self) -> None:
        last = self._cached[-1]
        last.terminated = tensor(True, device=self.device)
