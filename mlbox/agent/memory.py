import random
from collections import deque
from dataclasses import astuple, dataclass
from typing import Generic, TypeVar

import numpy as np

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

    def get_batch(self,
                  batch_size: int) -> tuple[tuple[T_State],
                                            tuple[T_Action],
                                            tuple[T_Reward],
                                            tuple[T_State],
                                            tuple[bool]]:
        size = min(len(self._memory), batch_size)
        samples = random.sample(self._memory, size)
        # batch = np.array(
        #     tuple(map(lambda exp: exp.tuple(), samples))
        # ).transpose()
        # states, actions, rewards, next_states, dones = batch
        # # states, next_states = np.stack(states), np.stack(next_states)
        states = tuple(s.state for s in samples)
        actions = tuple(s.action for s in samples)
        rewards = tuple(s.reward for s in samples)
        next_states = tuple(s.next_state for s in samples)
        dones = tuple(s.done for s in samples)
        return states, actions, rewards, next_states, dones
