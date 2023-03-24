from typing import Any, TypeVar

from mlbox.agent import Agent

T_State = TypeVar('T_State')
T_Action = TypeVar('T_Action')


class DQNAgent(Agent[T_State, T_Action]):
    def __init__(self,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
