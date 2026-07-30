"""
====================================================
QuantAI Professional v3.0
Module: data_loader.py
====================================================
Professional Binance Data Loader
====================================================
"""

from __future__ import annotations

from typing import Optional

import ccxt
import pandas as pd

from config.settings import (
    EXCHANGE,
    SYMBOL,
    TIMEFRAME,
    LIMIT,
)


# =====================================================
# Exchange Factory
# =====================================================

def create_exchange():

    exchange_name = EXCHANGE.lower()

    if exchange_name == "binance":

        exchange = ccxt.binance(
            {
                "enableRateLimit": True,
            }
        )

    else:

        raise ValueError(
            f"Unsupported exchange: {EXCHANGE}"
        )

    exchange.load_markets()

    return exchange


# =====================================================
# Download OHLCV
# =====================================================

def fetch_ohlcv(
    exchange,
    symbol: str,
    timeframe: str,
    limit: int,
):

    candles = exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )

    if candles is None:

        raise RuntimeError(
            "Exchange returned empty data."
        )

    if len(candles) == 0:

        raise RuntimeError(
            "No candles received."
        )

    return candles


# =====================================================
# Convert to DataFrame
# =====================================================

def candles_to_dataframe(
    candles,
) -> pd.DataFrame:

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna()

    df = df.reset_index(
        drop=True,
    )

    return df


# =====================================================
# Validation
# =====================================================

def validate_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in required:

        if column not in df.columns:

            raise RuntimeError(
                f"Column '{column}' not found."
            )

    if len(df) == 0:

        raise RuntimeError(
            "DataFrame is empty."
        )

    return df


# =====================================================
# Main Loader
# =====================================================

def load_binance_data(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:

    if symbol is None:
        symbol = SYMBOL

    if timeframe is None:
        timeframe = TIMEFRAME

    if limit is None:
        limit = LIMIT

    exchange = create_exchange()

    candles = fetch_ohlcv(
        exchange,
        symbol,
        timeframe,
        limit,
    )

    df = candles_to_dataframe(
        candles,
    )

    df = validate_dataframe(
        df,
    )

    return df


# =====================================================
# Last Candle
# =====================================================

def latest_candle(
    df: pd.DataFrame,
):

    return df.iloc[-1]


# =====================================================
# Symbol Information
# =====================================================

def market_info(
    symbol: str = SYMBOL,
):

    exchange = create_exchange()

    return exchange.market(symbol)


# =====================================================
# Available Symbols
# =====================================================

def available_symbols():

    exchange = create_exchange()

    return sorted(
        exchange.symbols
    )


# =====================================================
# Export
# =====================================================

__all__ = [

    "create_exchange",

    "fetch_ohlcv",

    "candles_to_dataframe",

    "validate_dataframe",

    "load_binance_data",

    "latest_candle",

    "market_info",

    "available_symbols",

]