"""
====================================================
QuantAI Trade Engine Diagnostic
Live Binance Data Test
====================================================
"""

from __future__ import annotations

import pandas as pd

from config.settings import (
    SYMBOL,
    TIMEFRAME,
    LIMIT,
    INITIAL_BALANCE,
)

from src.data_loader import load_binance_data
from src.indicators import add_indicators
from src.trade_engine import TradeEngine


# ====================================================
# HEADER
# ====================================================

print("=" * 60)
print("QUANTAI TRADE ENGINE DIAGNOSTIC")
print("=" * 60)


# ====================================================
# LOAD MARKET DATA
# ====================================================

print()
print("Loading Binance data...")

print(f"Symbol    : {SYMBOL}")
print(f"Timeframe : {TIMEFRAME}")
print(f"Limit     : {LIMIT}")

print()

df = load_binance_data(
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    limit=LIMIT,
)

print(f"Rows loaded : {len(df)}")


# ====================================================
# INDICATORS
# ====================================================

print()
print("Calculating indicators...")

df = add_indicators(df)

print("Indicators calculated.")


# ====================================================
# BASIC VALIDATION
# ====================================================

required_columns = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "atr",
    "ema_fast",
    "ema_slow",
    "ema_trend",
    "rsi",
    "macd",
    "macd_signal",
    "adx",
    "vwap",
    "trend",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    print()
    print("ERROR: Missing columns:")

    for column in missing:
        print(f" - {column}")

    raise SystemExit(1)


print()
print("Data validation: OK")


# ====================================================
# CREATE ENGINE
# ====================================================

print()
print("Creating Trade Engine...")

engine = TradeEngine()


print()
print("Initial Balance:")
print(f"{engine.balance:.2f}")


print()
print("Maximum Open Positions:")
print(engine.can_open_position())


# ====================================================
# RUN ENGINE
# ====================================================

print()
print("=" * 60)
print("RUNNING TRADE ENGINE")
print("=" * 60)

trades = engine.run(df)


# ====================================================
# REPORT
# ====================================================

print()
print("=" * 60)
print("TRADE ENGINE RESULT")
print("=" * 60)

print(
    f"Initial Balance : "
    f"{INITIAL_BALANCE:.2f}"
)

print(
    f"Final Balance   : "
    f"{engine.balance:.2f}"
)

print(
    f"Total Trades    : "
    f"{engine.total_trades}"
)

print(
    f"Wins            : "
    f"{engine.winning_trades}"
)

print(
    f"Losses          : "
    f"{engine.losing_trades}"
)

print(
    f"Win Rate        : "
    f"{engine.win_rate:.2f}%"
)

print(
    f"Total Profit    : "
    f"{engine.total_profit:.2f}"
)

print("=" * 60)


# ====================================================
# TRADES
# ====================================================

if len(trades) > 0:

    print()
    print("TRADES")
    print("-" * 60)

    print(
        trades.to_string(
            index=False
        )
    )

else:

    print()
    print("No trades generated.")


# ====================================================
# LAST CANDLE
# ====================================================

print()
print("=" * 60)
print("LAST CANDLE")
print("=" * 60)

row = df.iloc[-1]

print(f"Time        : {row['timestamp']}")
print(f"Open        : {row['open']:.2f}")
print(f"High        : {row['high']:.2f}")
print(f"Low         : {row['low']:.2f}")
print(f"Close       : {row['close']:.2f}")
print(f"Volume      : {row['volume']:.4f}")

print()

print(f"EMA Fast    : {row['ema_fast']:.2f}")
print(f"EMA Slow    : {row['ema_slow']:.2f}")
print(f"EMA Trend   : {row['ema_trend']:.2f}")
print(f"RSI         : {row['rsi']:.2f}")
print(f"MACD        : {row['macd']:.4f}")
print(f"MACD Signal : {row['macd_signal']:.4f}")
print(f"ATR         : {row['atr']:.2f}")
print(f"ADX         : {row['adx']:.2f}")
print(f"VWAP        : {row['vwap']:.2f}")
print(f"Trend       : {row['trend']}")

print("=" * 60)


# ====================================================
# FINISHED
# ====================================================

print()
print("=" * 60)
print("TRADE ENGINE DIAGNOSTIC FINISHED")
print("=" * 60)