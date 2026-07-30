import pandas as pd
import numpy as np


def calculate_statistics(trades: pd.DataFrame, initial_balance: float):
    """
    Рассчитывает основные показатели эффективности стратегии.
    """

    if trades.empty:
        return {
            "final_balance": initial_balance,
            "net_profit": 0.0,
            "roi": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
        }

    final_balance = trades.iloc[-1]["balance"]

    net_profit = final_balance - initial_balance

    roi = (net_profit / initial_balance) * 100

    wins = trades[trades["profit"] > 0]

    losses = trades[trades["profit"] <= 0]

    total_trades = len(trades)

    win_rate = len(wins) / total_trades * 100

    gross_profit = wins["profit"].sum()

    gross_loss = abs(losses["profit"].sum())

    if gross_loss == 0:
        profit_factor = np.inf
    else:
        profit_factor = gross_profit / gross_loss

    average_win = wins["profit"].mean() if len(wins) else 0

    average_loss = losses["profit"].mean() if len(losses) else 0

    expectancy = trades["profit"].mean()

    # ==========================
    # Max Drawdown
    # ==========================

    equity = trades["balance"]

    running_max = equity.cummax()

    drawdown = (equity - running_max) / running_max

    max_drawdown = drawdown.min() * 100

    return {

        "final_balance": final_balance,

        "net_profit": net_profit,

        "roi": roi,

        "trades": total_trades,

        "wins": len(wins),

        "losses": len(losses),

        "win_rate": win_rate,

        "profit_factor": profit_factor,

        "average_win": average_win,

        "average_loss": average_loss,

        "expectancy": expectancy,

        "max_drawdown": max_drawdown,

    }