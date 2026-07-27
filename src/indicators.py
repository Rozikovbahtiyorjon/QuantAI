import ta


def add_indicators(df):
    """
    Добавляет технические индикаторы в DataFrame.
    """

    # EMA
    df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)

    # RSI
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)

    # MACD
    df["macd"] = ta.trend.macd(df["close"])
    df["macd_signal"] = ta.trend.macd_signal(df["close"])

    # Bollinger Bands
    df["bb_upper"] = ta.volatility.bollinger_hband(df["close"])
    df["bb_lower"] = ta.volatility.bollinger_lband(df["close"])

    # ATR
    df["atr"] = ta.volatility.average_true_range(
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )

    return df