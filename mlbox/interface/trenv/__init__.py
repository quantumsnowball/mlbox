from gymnasium import Env
from pandas import Timestamp
from trbox.common.types import Symbol
from trbox.market.yahoo.historical.windows import YahooHistoricalWindows

from mlbox.interface.trenv.routine import Routine
from mlbox.types import T_Action, T_Obs


class TrEnv(
    Routine[T_Obs, T_Action],
    Env[T_Obs, T_Action],
):
    '''
    The interface for TrEnv
    '''
    Market: type[YahooHistoricalWindows]
    interval: int
    symbol: Symbol
    start: Timestamp | str
    end: Timestamp | str
    length: int

    ...
