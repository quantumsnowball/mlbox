from functools import wraps
from typing import Any, Callable, TypeVar

Prop = TypeVar('Prop')


def assured(prop: Callable[[Any], Prop]) -> Callable[[Any], Prop]:
    @wraps(prop)
    def wrapper(*args: Any, **kwargs: Any) -> Prop:
        try:
            return prop(*args, **kwargs)
        except AttributeError:
            raise NotImplementedError(prop.__name__) from None
    return wrapper
