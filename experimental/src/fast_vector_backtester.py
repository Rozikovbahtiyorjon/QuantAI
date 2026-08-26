from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class FastVectorBacktestResult:
    initial_balance: float
    final_balance: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    max_drawdown: float
    max_drawdown_percent: float
    equity_curve: list[float]


class FastVectorBacktester:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        commission: float = 0.0004,
        quantity: float = 1.0,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError(
                "initial_balance must be greater than zero."
            )

        if commission < 0:
            raise ValueError(
                "commission cannot be negative."
            )

        if quantity <= 0:
            raise ValueError(
                "quantity must be greater than zero."
            )

        self.initial_balance = float(initial_balance)
        self.commission = float(commission)
        self.quantity = float(quantity)

        self._result: Optional[
            FastVectorBacktestResult
        ] = None

    def run(
        self,
        df: pd.DataFrame,
    ) -> FastVectorBacktestResult:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "DataFrame cannot be empty."
            )

        required_columns = {
            "close",
            "signal",
        }

        missing = required_columns.difference(
            df.columns
        )

        if missing:
            raise ValueError(
                "DataFrame is missing required columns: "
                + ", ".join(sorted(missing))
            )

        prices = pd.to_numeric(
            df["close"],
            errors="coerce",
        )

        signals = (
            df["signal"]
            .astype(str)
            .str.upper()
        )

        if prices.isna().any():
            raise ValueError(
                "close contains invalid values."
            )

        if (prices <= 0).any():
            raise ValueError(
                "close must contain positive values."
            )

        position = 0
        entry_price = 0.0
        balance = self.initial_balance

        total_trades = 0
        winning_trades = 0
        losing_trades = 0

        equity_curve = [
            round(balance, 8)
        ]

        prices_array = prices.to_numpy(
            dtype=float
        )

        signals_array = signals.to_numpy()

        for price, signal in zip(
            prices_array,
            signals_array,
        ):
            price = float(price)

            if signal == "BUY":
                if position == 0:
                    position = 1
                    entry_price = price

            elif signal == "SELL":
                if position == 0:
                    position = -1
                    entry_price = price

                elif position == 1:
                    gross_profit = (
                        price - entry_price
                    ) * self.quantity

                    entry_notional = (
                        entry_price
                        * self.quantity
                    )

                    exit_notional = (
                        price
                        * self.quantity
                    )

                    commission_cost = (
                        entry_notional
                        + exit_notional
                    ) * self.commission

                    net_profit = (
                        gross_profit
                        - commission_cost
                    )

                    balance += net_profit

                    total_trades += 1

                    if net_profit > 0:
                        winning_trades += 1
                    elif net_profit < 0:
                        losing_trades += 1

                    equity_curve.append(
                        round(balance, 8)
                    )

                    position = -1
                    entry_price = price

            elif signal == "CLOSE":
                if position != 0:
                    gross_profit = (
                        (
                            price - entry_price
                        )
                        * self.quantity
                        * position
                    )

                    entry_notional = (
                        entry_price
                        * self.quantity
                    )

                    exit_notional = (
                        price
                        * self.quantity
                    )

                    commission_cost = (
                        entry_notional
                        + exit_notional
                    ) * self.commission

                    net_profit = (
                        gross_profit
                        - commission_cost
                    )

                    balance += net_profit

                    total_trades += 1

                    if net_profit > 0:
                        winning_trades += 1
                    elif net_profit < 0:
                        losing_trades += 1

                    position = 0
                    entry_price = 0.0

                    equity_curve.append(
                        round(balance, 8)
                    )

        if position != 0:
            final_price = float(
                prices_array[-1]
            )

            gross_profit = (
                (
                    final_price - entry_price
                )
                * self.quantity
                * position
            )

            entry_notional = (
                entry_price
                * self.quantity
            )

            exit_notional = (
                final_price
                * self.quantity
            )

            commission_cost = (
                entry_notional
                + exit_notional
            ) * self.commission

            net_profit = (
                gross_profit
                - commission_cost
            )

            balance += net_profit

            total_trades += 1

            if net_profit > 0:
                winning_trades += 1
            elif net_profit < 0:
                losing_trades += 1

            position = 0
            entry_price = 0.0

            equity_curve.append(
                round(balance, 8)
            )

        if total_trades > 0:
            win_rate = round(
                (
                    winning_trades
                    / total_trades
                )
                * 100.0,
                2,
            )
        else:
            win_rate = 0.0

        total_profit = round(
            balance - self.initial_balance,
            8,
        )

        total_return = round(
            (
                total_profit
                / self.initial_balance
            )
            * 100.0,
            2,
        )

        price_peak = float(
            prices_array[0]
        )

        max_drawdown = 0.0

        for price in prices_array:
            price = float(price)

            if price > price_peak:
                price_peak = price

            drawdown = (
                price_peak - price
            )

            if drawdown > max_drawdown:
                max_drawdown = drawdown

        equity_peak = max(
            equity_curve
        )

        if equity_peak > 0:
            max_drawdown_percent = round(
                (
                    max_drawdown
                    / equity_peak
                )
                * 100.0,
                6,
            )
        else:
            max_drawdown_percent = 0.0

        result = FastVectorBacktestResult(
            initial_balance=round(
                self.initial_balance,
                8,
            ),
            final_balance=round(
                balance,
                8,
            ),
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            max_drawdown=round(
                max_drawdown,
                8,
            ),
            max_drawdown_percent=(
                max_drawdown_percent
            ),
            equity_curve=equity_curve,
        )

        self._result = result

        return result

    @property
    def result(
        self,
    ) -> FastVectorBacktestResult | None:
        return self._result

    def reset(self) -> None:
        self._result = None


__all__ = [
    "FastVectorBacktestResult",
    "FastVectorBacktester",
]