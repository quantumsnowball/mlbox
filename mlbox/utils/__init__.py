from pandas import Series


def pnl_ratio(win: Series) -> float:
    pnlr = Series(win.rank(pct=True))
    return float(pnlr[-1])


def crop(value: float,
         low: float,
         high: float) -> float:
    ''' clip a value between low and high '''
    return min(max(value, low), high)
