from typing import Any

from trbox.event import Event
from trbox.event.system import Exit
from trbox.strategy import Strategy

from mlbox.trenv import TrEnv
from mlbox.trenv.queue import Terminated


class TrEnvStrategy(Strategy):
    def __init__(self,
                 *args: Any,
                 trenv: TrEnv,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trenv = trenv

    def handle(self, e: Event) -> None:
        super().handle(e)

        if isinstance(e, Exit):
            self.trenv.obs_q.put(Terminated())
            self.trenv.action_q.put(Terminated())
            self.trenv.reward_q.put(Terminated())
