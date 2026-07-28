import pandas as pd

from src.strategy import generate_signal


def run_backtest(
    df: pd.DataFrame,
    initial_balance=1000
):
    """
    Простейший backtest.
    Покупаем по сигналу BUY.
    Продаем по следующему сигналу SELL.
    """

    balance = initial_balance

    position = 0.0

    buy_price = 0.0

    trades = 0
    wins = 0
    losses = 0

    for i in range(60, len(df)):

        current_df = df.iloc[:i + 1]

        signal = generate_signal(current_df)

        price = current_df.iloc[-1]["close"]

        # ------------------------
        # BUY
        # ------------------------

        if signal == "BUY" and position == 0:

            position = balance / price

            buy_price = price

            balance = 0

            trades += 1

        # ------------------------
        # SELL
        # ------------------------

        elif signal == "SELL" and position > 0:

            balance = position * price

            if price > buy_price:
                wins += 1
            else:
                losses += 1

            position = 0

    # Если позиция осталась открытой
    if position > 0:

        last_price = df.iloc[-1]["close"]

        balance = position * last_price

    profit = balance - initial_balance

    profit_percent = (
        profit / initial_balance
    ) * 100

    print("\n" + "=" * 50)
    print("BACKTEST REPORT")
    print("=" * 50)

    print(f"Стартовый баланс : {initial_balance:.2f} $")
    print(f"Конечный баланс  : {balance:.2f} $")
    print(f"Прибыль          : {profit:.2f} $")
    print(f"Доходность       : {profit_percent:.2f}%")

    print()

    print(f"Сделок           : {trades}")
    print(f"Прибыльных       : {wins}")
    print(f"Убыточных        : {losses}")

    if trades > 0:

        win_rate = wins / trades * 100

        print(f"Win Rate         : {win_rate:.2f}%")