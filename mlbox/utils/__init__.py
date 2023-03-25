from pandas import Series


def pnl_ratio(win: Series) -> Series:
    return Series(win.rank(pct=True))


def crop(value: float,
         low: float,
         high: float) -> float:
    ''' clip a value between low and high '''
    return min(max(value, low), high)
