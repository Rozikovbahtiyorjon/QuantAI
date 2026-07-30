"""
====================================================
QuantAI Professional v3.0
Main Application
====================================================
"""

from __future__ import annotations

from config.settings import *

from src.data_loader import load_binance_data

from src.indicators import add_indicators

from src.strategy import (
    generate_signal_result,
    print_signal,
)

from src.risk_manager import (
    calculate_position_size,
)

from src.backtest import run_backtest


# =====================================================
# Header
# =====================================================

def print_header():

    print()

    print("=" * 60)

    print(PROJECT_NAME)

    print("=" * 60)

    print(f"Version      : {VERSION}")

    print(f"Exchange     : {EXCHANGE}")

    print(f"Symbol       : {SYMBOL}")

    print(f"Timeframe    : {TIMEFRAME}")

    print(f"Candles      : {LIMIT}")

    print("=" * 60)

    print()


# =====================================================
# Market Data
# =====================================================

def load_market():

    print("Loading Binance data...")

    df = load_binance_data(

        symbol=SYMBOL,

        timeframe=TIMEFRAME,

        limit=LIMIT,

    )

    print(f"Downloaded candles : {len(df)}")

    print()

    return df

    # =====================================================
# Indicator Engine
# =====================================================

def prepare_market(df):

    print("Calculating indicators...")

    df = add_indicators(df)

    print("Indicators calculated.")

    print()

    return df


# =====================================================
# Strategy
# =====================================================

def analyze_market(df):

    print("Running strategy engine...")

    result = generate_signal_result(df)

    print_signal(result)

    return result


# =====================================================
# Risk Management
# =====================================================

def calculate_trade(df, result):

    print()

    print("=" * 60)

    print("RISK MANAGEMENT")

    print("=" * 60)

    position_size = calculate_position_size(

        balance=INITIAL_BALANCE,

        risk_percent=RISK_PERCENT,

        entry_price=result.entry,

        stop_loss=result.stop_loss,

    )

    print(f"Balance         : {INITIAL_BALANCE:.2f} USDT")

    print(f"Risk            : {RISK_PERCENT:.2f}%")

    print(f"Entry           : {result.entry:.2f}")

    print(f"Stop Loss       : {result.stop_loss:.2f}")

    print(f"Take Profit     : {result.take_profit:.2f}")

    print(f"Position Size   : {position_size:.6f}")

    print("=" * 60)

    return position_size

    # =====================================================
# Last Candle
# =====================================================

def print_last_candle(df):

    print()

    print("=" * 60)

    print("LAST MARKET DATA")

    print("=" * 60)

    row = df.iloc[-1]

    print(f"Time            : {row['timestamp']}")

    print(f"Open            : {row['open']:.2f}")

    print(f"High            : {row['high']:.2f}")

    print(f"Low             : {row['low']:.2f}")

    print(f"Close           : {row['close']:.2f}")

    print(f"Volume          : {row['volume']:.4f}")

    print()

    print(f"EMA Fast        : {row['ema_fast']:.2f}")

    print(f"EMA Slow        : {row['ema_slow']:.2f}")

    print(f"EMA Trend       : {row['ema_trend']:.2f}")

    print(f"RSI             : {row['rsi']:.2f}")

    print(f"MACD            : {row['macd']:.4f}")

    print(f"MACD Signal     : {row['macd_signal']:.4f}")

    print(f"MACD Histogram  : {row['macd_hist']:.4f}")

    print(f"ATR             : {row['atr']:.2f}")

    print(f"ADX             : {row['adx']:.2f}")

    print(f"VWAP            : {row['vwap']:.2f}")

    print("=" * 60)


# =====================================================
# Backtest
# =====================================================

def run_statistics(df):

    print()

    print("=" * 60)

    print("STARTING BACKTEST")

    print("=" * 60)

    print()

    run_backtest(df)

    print()

    print("=" * 60)

    print("BACKTEST FINISHED")

    print("=" * 60)

    print()

    # =====================================================
# Main
# =====================================================

def main():

    print_header()

    df = load_market()

    df = prepare_market(df)

    result = analyze_market(df)

    calculate_trade(
        df,
        result,
    )

    print_last_candle(df)

    run_statistics(df)


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()

        print("=" * 60)

        print("Program interrupted by user.")

        print("=" * 60)

    except Exception as e:

        print()

        print("=" * 60)

        print("UNEXPECTED ERROR")

        print("=" * 60)

        print(type(e).__name__)

        print(str(e))

        print("=" * 60)

        raise