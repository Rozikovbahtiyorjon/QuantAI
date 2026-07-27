import pandas as pd

from src.strategy import generate_signal


def run_backtest(df: pd.DataFrame):
    """
    Простая проверка стратегии.
    """

    buy = 0
    sell = 0
    hold = 0

    # Начинаем после появления всех индикаторов
    for i in range(60, len(df)):

        signal = generate_signal(df.iloc[:i])

        if signal == "BUY":
            buy += 1

        elif signal == "SELL":
            sell += 1

        else:
            hold += 1

    print("\n========== BACKTEST ==========")
    print(f"BUY  : {buy}")
    print(f"SELL : {sell}")
    print(f"HOLD : {hold}")