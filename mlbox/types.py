from typing import TypeVar

from numpy.typing import ArrayLike

T_Obs = TypeVar('T_Obs', bound=ArrayLike)
T_Action = TypeVar('T_Action')
