from queue import Queue
from typing import Any, TypeVar

from typing_extensions import override


class Terminated:
    pass


class TerminatedError(Exception):
    pass


T = TypeVar('T')


class TrEnvQueue(Queue[T]):
    @override
    def get(self,
            *args: Any,
            **kwargs: Any) -> T:
        item = super().get(*args, **kwargs)
        if isinstance(item, Terminated):
            raise TerminatedError
        else:
            return item
