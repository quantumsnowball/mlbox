from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from trbox.event import Event
from trbox.event.system import Exit
from trbox.strategy import Strategy

from mlbox.trenv.queue import Terminated

if TYPE_CHECKING:
    from mlbox.trenv import TrEnv

from mlbox.types import T_Action, T_Obs


class TrEnvStrategy(Strategy, Generic[T_Obs, T_Action]):
    def __init__(self,
                 *args: Any,
                 trenv: TrEnv[T_Obs, T_Action],
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trenv = trenv

    def handle(self, e: Event) -> None:
        super().handle(e)

        if isinstance(e, Exit):
            self.trenv.obs_q.put(Terminated())
            self.trenv.action_q.put(Terminated())
            self.trenv.reward_q.put(Terminated())
