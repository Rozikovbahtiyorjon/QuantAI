from config.settings import *
from src.data_loader import load_binance_data
from src.indicators import add_indicators
from src.strategy import generate_signal
from src.backtest import run_backtest

print("=" * 50)
print(PROJECT_NAME)
print("=" * 50)

print(f"Биржа: {EXCHANGE}")
print(f"Пара: {SYMBOL}")
print(f"Таймфрейм: {TIMEFRAME}")

print("\nЗагрузка данных Binance...\n")

# ==========================================
# Загрузка исторических данных
# ==========================================

df = load_binance_data(
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    limit=200
)

print(f"Количество свечей: {len(df)}")

# ==========================================
# Добавляем технические индикаторы
# ==========================================

df = add_indicators(df)

# ==========================================
# Генерируем торговый сигнал
# ==========================================

signal = generate_signal(df)

print("\n" + "=" * 50)
print(f"Торговый сигнал: {signal}")
print("=" * 50)

# ==========================================
# Показываем текущие значения индикаторов
# ==========================================

last = df.iloc[-1]

print("\nТекущие показатели:")

print(f"Цена         : {last['close']:.2f}")
print(f"EMA20        : {last['ema20']:.2f}")
print(f"EMA50        : {last['ema50']:.2f}")
print(f"RSI          : {last['rsi']:.2f}")
print(f"MACD         : {last['macd']:.4f}")
print(f"MACD Signal  : {last['macd_signal']:.4f}")
print(f"ATR          : {last['atr']:.2f}")

# ==========================================
# Последняя свеча
# ==========================================

print("\nПоследняя свеча:\n")
print(df.tail(1))

# ==========================================
# Запуск простого Backtest
# ==========================================

run_backtest(df)