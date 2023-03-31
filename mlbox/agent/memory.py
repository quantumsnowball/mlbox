from collections import deque
from dataclasses import asdict, astuple, dataclass
from typing import Generic, SupportsFloat

import numpy as np
import numpy.typing as npt
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

    def __post_init__(self):
        self.reward = np.float32(self.reward)


# @dataclass
# class Batch:
#     obs: npt.NDArray[np.float32]
#     action: npt.NDArray[np.float32]
#     reward: npt.NDArray[np.float32]
#     next_obs: npt.NDArray[np.float32]
#     terminated: npt.NDArray[np.bool8]


class Replay(Dataset[Experience[T_Obs, T_Action]],
             Generic[T_Obs, T_Action]):
    def __init__(self,
                 maxlen: int = 10000):
        self._memory = deque[Experience[T_Obs,
                                        T_Action]](maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._memory)

    def __getitem__(self,
                    index: int) -> dict:
        return asdict(self._memory[index])

    def remember(self,
                 exp: Experience[T_Obs,
                                 T_Action]) -> None:
        self._memory.append(exp)

    def sample(self,
               batch_size: int) -> dict:
        def collate(batch: list[dict]) -> dict:
            # avoid create tensor from list of nd.array
            def to_tensor(k: str) -> Tensor:
                column = [b[k] for b in batch]
                return tensor(np.stack(column))
            keys = tuple(batch[0].keys())
            tensor_dict = {k: to_tensor(k) for k in keys}
            return tensor_dict

        # sampling
        loader = DataLoader(self, batch_size, shuffle=True, collate_fn=collate)
        # pack
        samples = next(iter(loader))
        return samples
