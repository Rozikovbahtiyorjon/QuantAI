from config.settings import *
from src.data_loader import load_binance_data

print("=" * 50)
print(PROJECT_NAME)
print("=" * 50)

print(f"Биржа: {EXCHANGE}")
print(f"Пара: {SYMBOL}")
print(f"Таймфрейм: {TIMEFRAME}")

print("\nЗагрузка данных Binance...\n")

df = load_binance_data(
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    limit=10
)

print(df)