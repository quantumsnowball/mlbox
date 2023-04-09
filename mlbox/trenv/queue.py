from queue import Queue
from typing import Any, TypeVar

from typing_extensions import override

from mlbox.events import Terminated, TerminatedError

T = TypeVar('T')


class TrEnvQueue(Queue[T | Terminated]):
    @override
    def get(self,
            *args: Any,
            **kwargs: Any) -> T:
        item = super().get(*args, **kwargs)
        if isinstance(item, Terminated):
            raise TerminatedError
        else:
            return item

    def clear(self) -> None:
        self.queue.clear()
