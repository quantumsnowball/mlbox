from typing import Any, TypeVar

from mlbox.agent import Agent
from mlbox.agent.memory import Replay

T_State = TypeVar('T_State')
T_Action = TypeVar('T_Action')
T_Reward = TypeVar('T_Reward')


class DQNAgent(Agent[T_State, T_Action, T_Reward]):
    def __init__(self,
                 *args: Any,
                 replay_size: int = 10000,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._replay = Replay[T_State, T_Action, T_Reward](replay_size)
