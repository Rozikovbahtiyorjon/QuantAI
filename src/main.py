from config.settings import *
from src.data_loader import load_binance_data
from src.indicators import add_indicators
from src.strategy import generate_signal
from src.backtest import run_backtest
from src.risk_manager import (
    calculate_position_size,
    calculate_sl_tp
)

print("=" * 50)
print(PROJECT_NAME)
print("=" * 50)

print(f"Биржа: {EXCHANGE}")
print(f"Пара: {SYMBOL}")
print(f"Таймфрейм: {TIMEFRAME}")

print("\nЗагрузка данных Binance...\n")

# =====================================
# Загрузка исторических данных
# =====================================

df = load_binance_data(
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    limit=200
)

print(f"Количество свечей: {len(df)}")

# =====================================
# Добавление индикаторов
# =====================================

df = add_indicators(df)

# =====================================
# Получение торгового сигнала
# =====================================

signal = generate_signal(df)

print("\n" + "=" * 50)
print(f"Торговый сигнал: {signal}")
print("=" * 50)

# =====================================
# Последняя свеча
# =====================================

last = df.iloc[-1]

print("\nТекущие показатели рынка")

print(f"Цена          : {last['close']:.2f}")
print(f"EMA20         : {last['ema20']:.2f}")
print(f"EMA50         : {last['ema50']:.2f}")
print(f"RSI           : {last['rsi']:.2f}")
print(f"MACD          : {last['macd']:.4f}")
print(f"MACD Signal   : {last['macd_signal']:.4f}")
print(f"ATR           : {last['atr']:.2f}")

# =====================================
# Управление капиталом
# =====================================

balance = 1000
risk = 1

entry_price = last["close"]
atr = last["atr"]

stop_loss, take_profit = calculate_sl_tp(
    entry_price,
    atr
)

position_size = calculate_position_size(
    balance,
    risk,
    entry_price,
    stop_loss
)

print("\n" + "=" * 50)
print("Управление капиталом")
print("=" * 50)

print(f"Баланс            : {balance}$")
print(f"Риск              : {risk}%")
print(f"Цена входа        : {entry_price:.2f}")
print(f"Stop Loss         : {stop_loss}")
print(f"Take Profit       : {take_profit}")
print(f"Размер позиции    : {position_size} BTC")

# =====================================
# Последняя свеча
# =====================================

print("\nПоследняя свеча:")

print(df.tail(1))

# =====================================
# Backtest
# =====================================

run_backtest(df)