"""
=======================================================
QuantAI Professional v3.2
Indicators Module - Full Set with Core 4 Focus

All indicators available for ML feature engineering.
Strategy uses only Core 4: EMA, RSI, ATR, Volume Ratio
=======================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    EMA_FAST,
    EMA_SLOW,
    EMA_TREND,
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    ATR_PERIOD,
    ADX_PERIOD,
    ADX_MIN,
    BB_PERIOD,
    BB_STD,
    VOLUME_MA,
    VOLUME_FILTER,
    SUPERTREND_PERIOD,
    SUPERTREND_MULTIPLIER,
)

# ==========================================================
# MOVING AVERAGES
# ==========================================================

def sma(
    series: pd.Series,
    period: int,
) -> pd.Series:
    """
    Simple Moving Average.
    """

    return (
        series
        .rolling(
            window=period,
            min_periods=period,
        )
        .mean()
    )


def ema(
    series: pd.Series,
    period: int,
) -> pd.Series:
    """
    Exponential Moving Average.
    """

    return (
        series
        .ewm(
            span=period,
            adjust=False,
        )
        .mean()
    )


# ==========================================================
# RSI
# ==========================================================

def rsi(
    close: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:
    """
    Relative Strength Index (Wilder).
    """

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False,
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False,
        )
        .mean()
    )

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    value = 100 - (100 / (1 + rs))

    return value.fillna(50.0)


# ==========================================================
# MACD
# ==========================================================

def macd(
    close: pd.Series,
):
    """
    Moving Average Convergence Divergence.
    """

    ema_fast = ema(
        close,
        MACD_FAST,
    )

    ema_slow = ema(
        close,
        MACD_SLOW,
    )

    macd_line = ema_fast - ema_slow

    signal_line = ema(
        macd_line,
        MACD_SIGNAL,
    )

    histogram = macd_line - signal_line

    return (
        macd_line,
        signal_line,
        histogram,
    )


# ==========================================================
# TRUE RANGE
# ==========================================================

def true_range(
    df: pd.DataFrame,
) -> pd.Series:
    """
    True Range.
    """

    high_low = (
        df["high"] - df["low"]
    )

    high_close = (
        df["high"] - df["close"].shift()
    ).abs()

    low_close = (
        df["low"] - df["close"].shift()
    ).abs()

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close,
        ],
        axis=1,
    ).max(axis=1)

    return tr


# ==========================================================
# AVERAGE TRUE RANGE
# ==========================================================

def atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """
    Average True Range (Wilder EMA).
    """

    tr = true_range(df)

    return (
        tr
        .ewm(
            alpha=1 / period,
            adjust=False,
        )
        .mean()
    )


# ==========================================================
# ADX
# ==========================================================

def adx(
    df: pd.DataFrame,
    period: int = ADX_PERIOD,
):
    """
    Average Directional Index.
    Returns:
        plus_di,
        minus_di,
        adx
    """

    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) &
            (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) &
            (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    atr_value = atr(
        df,
        period,
    )

    plus_di = (
        100
        *
        plus_dm
        .ewm(
            alpha=1 / period,
            adjust=False,
        )
        .mean()
        /
        atr_value
    )

    minus_di = (
        100
        *
        minus_dm
        .ewm(
            alpha=1 / period,
            adjust=False,
        )
        .mean()
        /
        atr_value
    )

    dx = (
        (
            plus_di
            -
            minus_di
        ).abs()
        /
        (
            plus_di
            +
            minus_di
        ).replace(
            0,
            np.nan,
        )
    ) * 100

    adx_line = (
        dx
        .ewm(
            alpha=1 / period,
            adjust=False,
        )
        .mean()
    )

    return (
        plus_di.fillna(0.0),
        minus_di.fillna(0.0),
        adx_line.fillna(0.0),
    )


# ==========================================================
# BOLLINGER BANDS
# ==========================================================

def bollinger(
    close: pd.Series,
    period: int = BB_PERIOD,
    std: float = BB_STD,
):
    """
    Bollinger Bands.
    """

    middle = sma(
        close,
        period,
    )

    deviation = (
        close
        .rolling(
            window=period,
            min_periods=period,
        )
        .std()
    )

    upper = middle + deviation * std
    lower = middle - deviation * std

    return (
        upper,
        middle,
        lower,
    )


# ==========================================================
# VWAP
# ==========================================================

def vwap(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Volume Weighted Average Price.
    """

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    cumulative_volume = df["volume"].cumsum()

    cumulative_price_volume = (
        typical_price
        * df["volume"]
    ).cumsum()

    return (
        cumulative_price_volume
        /
        cumulative_volume
    )


# ==========================================================
# ON BALANCE VOLUME
# ==========================================================

def obv(
    df: pd.DataFrame,
) -> pd.Series:
    """
    On Balance Volume.
    """

    direction = np.sign(
        df["close"].diff()
    ).fillna(0.0)

    return (
        direction
        * df["volume"]
    ).cumsum()


# ==========================================================
# VOLUME MOVING AVERAGE
# ==========================================================

def volume_sma(
    volume: pd.Series,
    period: int = VOLUME_MA,
) -> pd.Series:
    """
    Volume SMA.
    """

    return (
        volume
        .rolling(
            window=period,
            min_periods=period,
        )
        .mean()
    )


# ==========================================================
# SUPERTREND
# ==========================================================

def supertrend(
    df: pd.DataFrame,
    period: int = SUPERTREND_PERIOD,
    multiplier: float = SUPERTREND_MULTIPLIER,
):
    """
    SuperTrend.

    Returns
    -------
    supertrend : pd.Series
    trend      : pd.Series
                 1  = Bull Trend
                -1  = Bear Trend
    """

    atr_value = atr(
        df,
        period,
    )

    hl2 = (
        df["high"]
        +
        df["low"]
    ) / 2.0

    upper_band = hl2 + multiplier * atr_value
    lower_band = hl2 - multiplier * atr_value

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    trend = pd.Series(
        index=df.index,
        dtype="int8",
    )

    supertrend_line = pd.Series(
        index=df.index,
        dtype="float64",
    )

    trend.iloc[0] = 1
    supertrend_line.iloc[0] = lower_band.iloc[0]

    for i in range(1, len(df)):

        if (
            upper_band.iloc[i] < final_upper.iloc[i - 1]
            or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (
            lower_band.iloc[i] > final_lower.iloc[i - 1]
            or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if trend.iloc[i - 1] == 1:

            if df["close"].iloc[i] < final_lower.iloc[i]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = 1

        else:

            if df["close"].iloc[i] > final_upper.iloc[i]:
                trend.iloc[i] = 1
            else:
                trend.iloc[i] = -1

        if trend.iloc[i] == 1:
            supertrend_line.iloc[i] = final_lower.iloc[i]
        else:
            supertrend_line.iloc[i] = final_upper.iloc[i]

    # Causal warmup: never backward-fill future values; drop warmup upstream via DatasetBuilder warmup_bars
    return (
        supertrend_line.ffill().fillna(lower_band.iloc[0] if len(lower_band) else 0.0),
        trend.fillna(1),
    )


# ==========================================================
# TREND SCORE
# ==========================================================

def trend_score(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Composite Trend Strength Score.
    """

    score = pd.Series(
        0.0,
        index=df.index,
        dtype="float64",
    )

    # EMA Alignment
    score += np.where(
        df["ema_fast"] > df["ema_slow"],
        1.0,
        -1.0,
    )

    score += np.where(
        df["ema_slow"] > df["ema_trend"],
        1.0,
        -1.0,
    )

    # RSI
    score += np.where(
        df["rsi"] > 60,
        0.5,
        np.where(
            df["rsi"] < 40,
            -0.5,
            0.0,
        ),
    )

    # MACD
    score += np.where(
        df["macd"] > df["macd_signal"],
        1.0,
        -1.0,
    )

    # ADX
    score += np.where(
        df["adx"] > ADX_MIN,
        0.5,
        0.0,
    )

    # SuperTrend
    score += np.where(
        df["trend"] == 1,
        1.0,
        -1.0,
    )

    return score


# ==========================================================
# VOLUME FILTER
# ==========================================================

def volume_filter(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Detect abnormal volume.
    """

    return (
        df["volume"]
        >
        df["volume_sma20"]
        *
        VOLUME_FILTER
    ).fillna(False)


# ==========================================================
# VOLATILITY FILTER
# ==========================================================

def volatility_filter(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Detect high volatility.
    """

    atr_mean = (
        df["atr"]
        .rolling(
            window=20,
            min_periods=20,
        )
        .mean()
    )

    return (
        df["atr"]
        >
        atr_mean
    ).fillna(False)


# ==========================================================
# BREAKOUT FILTER
# ==========================================================

def breakout_filter(
    df: pd.DataFrame,
    period: int = 20,
):
    """
    Price breakout detector.
    """

    highest = (
        df["high"]
        .rolling(
            window=period,
            min_periods=period,
        )
        .max()
    )

    lowest = (
        df["low"]
        .rolling(
            window=period,
            min_periods=period,
        )
        .min()
    )

    breakout_up = (
        df["close"]
        >
        highest.shift(1)
    ).fillna(False)

    breakout_down = (
        df["close"]
        <
        lowest.shift(1)
    ).fillna(False)

    return (
        breakout_up,
        breakout_down,
    )


# ==========================================================
# BUILD ALL INDICATORS (for ML feature engineering)
# ==========================================================

def add_indicators(
    df: pd.DataFrame,
    core_only: bool = False,
) -> pd.DataFrame:
    """
    Calculate indicators used by QuantAI.
    
    Args:
        df: OHLCV DataFrame
        core_only: If True, compute only Core 4 (EMA, RSI, ATR, Volume Ratio).
                   If False, compute all indicators for ML feature engineering.
    """

    df = df.copy()

    # ======================================================
    # CORE 4: EMA, RSI, ATR, Volume
    # ======================================================

    df["ema_fast"] = ema(
        df["close"],
        EMA_FAST,
    )

    df["ema_slow"] = ema(
        df["close"],
        EMA_SLOW,
    )

    df["ema_trend"] = ema(
        df["close"],
        EMA_TREND,
    )

    df["rsi"] = rsi(
        df["close"],
        RSI_PERIOD,
    )

    df["atr"] = atr(
        df,
        ATR_PERIOD,
    )

    df["volume_sma20"] = volume_sma(
        df["volume"],
        VOLUME_MA,
    )

    df["volume_sma"] = df["volume_sma20"]  # alias for feature_engine

    df["volume_ratio"] = df["volume"] / df["volume_sma20"].replace(0, np.nan)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

    if core_only:
        return _cleanup_dataframe(df)

    # ======================================================
    # EXTENDED INDICATORS (for ML)
    # ======================================================

    # MACD
    (
        df["macd"],
        df["macd_signal"],
        df["macd_hist"],
    ) = macd(
        df["close"],
    )

    # ADX
    (
        df["plus_di"],
        df["minus_di"],
        df["adx"],
    ) = adx(
        df,
        ADX_PERIOD,
    )

    # BOLLINGER
    (
        df["bb_upper"],
        df["bb_middle"],
        df["bb_lower"],
    ) = bollinger(
        df["close"],
        BB_PERIOD,
        BB_STD,
    )

    # Derived Bollinger features for Range mean-reversion strategy
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-9)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9) - 0.5  # -0.5..0.5
    df["bb_squeeze"] = (df["bb_width"] < 0.02).astype(float)

    # VWAP
    df["vwap"] = vwap(df)

    # OBV
    df["obv"] = obv(df)

    df["obv_ema"] = (
        df["obv"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    # SUPERTREND
    (
        df["supertrend"],
        df["trend"],
    ) = supertrend(
        df,
        SUPERTREND_PERIOD,
        SUPERTREND_MULTIPLIER,
    )

    # TREND SCORE
    df["trend_score"] = trend_score(df)

    # FILTERS
    df["volume_filter"] = volume_filter(df)
    df["volatility_filter"] = volatility_filter(df)
    (
        df["breakout_up"],
        df["breakout_down"],
    ) = breakout_filter(df)

    return _cleanup_dataframe(df)


def _cleanup_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up NaN/inf values — CAUSAL ONLY (P0 fix).

    NO bfill: forward-fill would inject future values into past rows
    (look-ahead bias). Warm-up NaNs are kept and dropped upstream
    by DatasetBuilder (warmup_bars) / BacktestEngine minimum_rows.
    """
    numeric_columns = df.select_dtypes(
        include=["number"],
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # Causal only: propagate last known value forward, never backward.
    df[numeric_columns] = df[numeric_columns].ffill()

    bool_columns = [
        "volume_filter",
        "volatility_filter",
        "breakout_up",
        "breakout_down",
    ]

    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].fillna(False)

    return df


# ==========================================================
# CORE 4 INDICATORS ONLY (for strategy) - DEPRECATED
# Use add_indicators(df, core_only=True) instead
# ==========================================================

def add_core_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate only the 4 core indicators:
    1. EMA (fast, slow, trend)
    2. RSI
    3. ATR
    4. Volume Ratio (volume / volume_sma)
    
    Deprecated: use add_indicators(df, core_only=True) instead.
    """
    return add_indicators(df, core_only=True)


# ==========================================================
# EXPORTS
# ==========================================================

__all__ = [
    "sma",
    "ema",
    "rsi",
    "macd",
    "true_range",
    "atr",
    "adx",
    "bollinger",
    "vwap",
    "obv",
    "volume_sma",
    "supertrend",
    "trend_score",
    "volume_filter",
    "volatility_filter",
    "breakout_filter",
    "add_indicators",
    "add_core_indicators",
]