from pandas import Series


def pnl_ratio(win: Series) -> float:
    pnlr = Series(win.rank(pct=True))
    return float(pnlr[-1])
