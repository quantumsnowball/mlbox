from typing import Any, TypeVar

from numpy.typing import NDArray

T_Obs = TypeVar('T_Obs', bound=NDArray[Any])
T_Action = TypeVar('T_Action', bound=NDArray[Any])
