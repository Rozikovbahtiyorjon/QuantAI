import pandas as pd

from src.trade_engine import run_trade_engine


def run_backtest(df: pd.DataFrame):
    """
    Запускает полный backtest стратегии.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame со свечами и индикаторами.

    Returns
    -------
    pandas.DataFrame
        Журнал всех сделок.
    """

    print()
    print("=" * 50)
    print("BACKTEST")
    print("=" * 50)

    trades = run_trade_engine(df)

    return trades