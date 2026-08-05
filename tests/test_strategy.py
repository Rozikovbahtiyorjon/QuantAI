"""
====================================================
QuantAI Professional v5.0
Strategy Test
====================================================
"""

from src.data_loader import load_binance_data
from src.indicators import add_indicators
from src.strategy import generate_signal_result


print("Loading data...")

df = load_binance_data()

print("Calculating indicators...")

df = add_indicators(df)

print("Generating signal...")

signal = generate_signal_result(df)

print()

print("=" * 60)
print("STRATEGY TEST")
print("=" * 60)

print(f"Signal      : {signal.signal}")
print(f"Confidence  : {signal.confidence:.2f}%")
print(f"Score       : {signal.score:.2f}")

print()

print(f"Entry       : {signal.entry:.2f}")
print(f"Stop Loss   : {signal.stop_loss:.2f}")
print(f"Take Profit : {signal.take_profit:.2f}")

print()

print("Reasons:")

for reason in signal.reasons:
    print(f" • {reason}")

print("=" * 60)