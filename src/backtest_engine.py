"""
=========================================================
QuantAI Professional v5
Backtest Engine

Historical strategy backtesting.

Pipeline:
    Prepared OHLCV + indicators
        в†“
    BacktestEngine
        в†“
    TradeEngine
        в†“
    Strategy

The Backtest Engine does NOT:
    - load market data
    - use CCXT
    - calculate indicators
    - modify Strategy
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.trade_engine import TradeEngine


# ==========================================================
# CONFIGURATION
# ==========================================================

MINIMUM_ROWS = 300

REQUIRED_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "atr",
}


# ==========================================================
# RESULT
# ==========================================================

@dataclass
class BacktestResult:
    """
    Final backtest statistics.
    """

    initial_balance: float
    final_balance: float
    net_profit: float

    total_trades: int
    winning_trades: int
    losing_trades: int

    win_rate: float

    trades: Any = None


# ==========================================================
# ENGINE
# ==========================================================

class BacktestEngine:
    """
    Historical backtesting engine.

    Receives a DataFrame that has already passed through
    src.indicators.add_indicators().
    """

    def __init__(
        self,
        initial_balance: float | None = None,
    ) -> None:

        self.trade_engine = TradeEngine()

        if initial_balance is not None:
            self._set_initial_balance(initial_balance)

    # ======================================================
    # INTERNAL BALANCE SETTER
    # ======================================================

    def _set_initial_balance(
        self,
        balance: float,
    ) -> None:

        # Support the existing TradeEngine balance naming.
        if hasattr(self.trade_engine, "initial_balance"):
            self.trade_engine.initial_balance = float(balance)

        if hasattr(self.trade_engine, "balance"):
            self.trade_engine.balance = float(balance)

        if hasattr(self.trade_engine, "equity"):
            self.trade_engine.equity = float(balance)

    # ======================================================
    # VALIDATION
    # ======================================================

    def validate_data(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate prepared historical data.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "BacktestEngine requires a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Backtest data is empty."
            )

        missing = REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(sorted(missing))
            )

        if len(df) < MINIMUM_ROWS:
            raise ValueError(
                f"Backtest requires at least "
                f"{MINIMUM_ROWS} rows. "
                f"Received: {len(df)}"
            )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
        ]

        for column in numeric_columns:

            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):
                raise TypeError(
                    f"Column '{column}' must be numeric."
                )

        if df[numeric_columns].isna().any().any():
            raise ValueError(
                "Backtest data contains NaN values "
                "in required numeric columns."
            )

    # ======================================================
    # ENGINE STAT EXTRACTION
    # ======================================================

    @staticmethod
    def _get_value(
        obj: Any,
        names: list[str],
        default: Any = 0,
    ) -> Any:

        for name in names:

            if hasattr(obj, name):
                value = getattr(obj, name)

                if value is not None:
                    return value

        return default

    # ======================================================
    # RUN
    # ======================================================

    def run(
        self,
        df: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run historical backtest.

        Indicators must already be calculated.
        """

        self.validate_data(df)

        data = df.copy()

        data = data.reset_index(drop=True)

        # --------------------------------------------------
        # Initial balance
        # --------------------------------------------------

        initial_balance = float(
            self._get_value(
                self.trade_engine,
                [
                    "initial_balance",
                    "starting_balance",
                    "balance",
                    "equity",
                ],
                0.0,
            )
        )

        # --------------------------------------------------
        # Run Trade Engine
        # --------------------------------------------------

        engine_result = self.trade_engine.run(data)

        # --------------------------------------------------
        # Extract final balance
        # --------------------------------------------------
        
        if isinstance(engine_result, pd.DataFrame):
        
            final_balance = float(
                getattr(
                    self.trade_engine,
                    "balance",
                    initial_balance,
                )
             )
        
        else:
        
            final_balance = float(
                self._get_value(
                    engine_result,
                    [
                        "final_balance",
                        "balance",
                        "equity",
                        "current_balance",
                    ], 
                    getattr(
                        self.trade_engine,
                        "balance",
                        initial_balance,
                    ),
                 )
            )
        
        # --------------------------------------------------
        # --------------------------------------------------
        # Net Profit
        # --------------------------------------------------

        if isinstance(engine_result, pd.DataFrame):

            if "net_profit" in engine_result.columns:
                net_profit = float(engine_result["net_profit"].sum())
            else:
                net_profit = float(final_balance - initial_balance)

        else:

            net_profit = float(
                self._get_value(
                    engine_result,
                    [
                        "net_profit",
                        "profit",
                        "total_profit",
                    ],
                    final_balance - initial_balance,
                )
            )


        total_trades = int(
            self._get_value(
                engine_result,
                [
                    "total_trades",
                    "trade_count",
                    "number_of_trades",
                ],
                self._get_value(
                    self.trade_engine,
                    [
                        "total_trades",
                        "trade_count",
                        "number_of_trades",
                    ],
                    0,
                ),
            )
        )

        winning_trades = int(
            self._get_value(
                engine_result,
                [
                    "winning_trades",
                    "wins",
                    "winning_count",
                ],
                self._get_value(
                    self.trade_engine,
                    [
                        "winning_trades",
                        "wins",
                        "winning_count",
                    ],
                    0,
                ),
            )
        )

        losing_trades = int(
            self._get_value(
                engine_result,
                [
                    "losing_trades",
                    "losses",
                    "losing_count",
                ],
                self._get_value(
                    self.trade_engine,
                    [
                        "losing_trades",
                        "losses",
                        "losing_count",
                    ],
                    0,
                ),
            )
        )

        if total_trades > 0:
            win_rate = (
                winning_trades
                / total_trades
                * 100.0
            )
        else:
            win_rate = 0.0

        # --------------------------------------------------
        # Trades
        # --------------------------------------------------

        if isinstance(engine_result, pd.DataFrame):

            trades = engine_result.to_dict(
                orient="records"
            )

        else:

            trades = self._get_value(
                engine_result,
                [
                    "trades",
                    "trade_history",
                    "history",
                ],
                self._get_value(
                    self.trade_engine,
                    [
                        "trades",
                        "trade_history",
                        "history",
                    ],
                    [],
                ),
            )

            if trades is None:
                trades = []

        return BacktestResult(
            initial_balance=initial_balance,
            final_balance=final_balance,
            net_profit=net_profit,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            trades=trades,
        )

    # ======================================================
    # REPORT
    # ======================================================

    @staticmethod
    def print_report(
        result: BacktestResult,
    ) -> None:

        print()
        print("=" * 60)
        print("QUANTAI BACKTEST REPORT")
        print("=" * 60)

        print(
            f"Initial Balance : "
            f"{result.initial_balance:.2f}"
        )

        print(
            f"Final Balance   : "
            f"{result.final_balance:.2f}"
        )

        print(
            f"Net Profit      : "
            f"{result.net_profit:.2f}"
        )

        print("-" * 60)

        print(
            f"Total Trades    : "
            f"{result.total_trades}"
        )

        print(
            f"Winning Trades  : "
            f"{result.winning_trades}"
        )

        print(
            f"Losing Trades   : "
            f"{result.losing_trades}"
        )

        print(
            f"Win Rate        : "
            f"{result.win_rate:.2f}%"
        )

        print("=" * 60)


# ==========================================================
# CONVENIENCE FUNCTION
# ==========================================================

def run_backtest(
    df: pd.DataFrame,
) -> BacktestResult:
    """
    Convenience wrapper.
    """

    engine = BacktestEngine()

    result = engine.run(df)

    engine.print_report(result)

    return result


# ==========================================================
# EXPORTS
# ==========================================================

__all__ = [
    "BacktestResult",
    "BacktestEngine",
    "run_backtest",
]
