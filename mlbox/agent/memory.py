from collections import deque
from dataclasses import dataclass
from typing import Any, Generic, SupportsFloat

import numpy as np
from numpy.typing import NDArray
from torch import Tensor, tensor
from torch.utils.data import DataLoader, Dataset

from mlbox.types import T_Action, T_Obs


@dataclass
class Experience(Generic[T_Obs,
                         T_Action]):
    obs: T_Obs
    action: T_Action
    reward: SupportsFloat
    next_obs: T_Obs
    terminated: bool

    def __post_init__(self) -> None:
        self.reward = np.float32(self.reward)


@dataclass
class Batch:
    obs: Tensor
    action: Tensor
    reward: Tensor
    next_obs: Tensor
    terminated: Tensor


class Replay(Dataset[Experience[T_Obs, T_Action]],
             Generic[T_Obs, T_Action]):
    def __init__(self,
                 maxlen: int = 10000):
        self._memory = deque[Experience[T_Obs,
                                        T_Action]](maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._memory)

    def __getitem__(self,
                    index: int) -> Experience[T_Obs,
                                              T_Action]:
        return self._memory[index]

    def remember(self,
                 exp: Experience[T_Obs,
                                 T_Action]) -> None:
        self._memory.append(exp)

    def sample(self,
               batch_size: int,
               *,
               device: str = 'cpu') -> Batch:
        def collate(batch: list[Experience[T_Obs, T_Action]]) -> Batch:
            # avoid create tensor from list of nd.array
            def to_tensor(arr: NDArray[Any]) -> Tensor:
                return tensor(arr, device=device)
            return Batch(
                obs=to_tensor(np.stack([b.obs for b in batch])),
                action=to_tensor(np.array([b.action for b in batch])),
                reward=to_tensor(np.array([b.reward for b in batch])),
                next_obs=to_tensor(np.stack([b.next_obs for b in batch])),
                terminated=to_tensor(np.stack([b.terminated for b in batch]))
            )

        # sampling
        loader = DataLoader[Experience[T_Obs, T_Action]](self,
                                                         batch_size,
                                                         shuffle=True,
                                                         collate_fn=collate)
        # pack
        samples: Batch = next(iter(loader))
        return samples
