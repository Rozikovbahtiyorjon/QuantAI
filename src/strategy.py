import pandas as pd


def generate_signal(df: pd.DataFrame):
    """
    Возвращает торговый сигнал:
    BUY
    SELL
    HOLD
    """

    last = df.iloc[-1]

    # Покупка
    if (
        last["ema20"] > last["ema50"]
        and last["macd"] > last["macd_signal"]
        and last["rsi"] < 70
    ):
        return "BUY"

    # Продажа
    if (
        last["ema20"] < last["ema50"]
        and last["macd"] < last["macd_signal"]
        and last["rsi"] > 30
    ):
        return "SELL"

    return "HOLD"