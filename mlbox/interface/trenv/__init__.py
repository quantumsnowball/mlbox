from gymnasium import Env

from mlbox.types import T_Action, T_Obs


class TrEnv(
    Env[T_Obs, T_Action],
):
    '''
    The interface for TrEnv
    '''
    ...
