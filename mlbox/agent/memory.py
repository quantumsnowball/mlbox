from collections import deque
from dataclasses import asdict, dataclass
from typing import Generic, SupportsFloat

import numpy as np
import numpy.typing as npt
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

    def tuple(self) -> tuple[T_Obs,
                             T_Action,
                             SupportsFloat,
                             T_Obs,
                             bool]:
        return (self.obs, self.action, self.reward, self.next_obs, self.terminated, )


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
        # sampling
        loader = DataLoader(self, batch_size, shuffle=True)
        # pack
        samples = next(iter(loader))
        return samples
