"""
=========================================================
QuantAI Professional v5
Backtest Engine

Historical strategy backtesting.

Pipeline:
    Prepared OHLCV + indicators
        ↓
    BacktestEngine
        ↓
    TradeEngine
        ↓
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

    Core fields are backward compatible; risk metrics
    (MetricsVector) were added in Phase 1.
    """

    initial_balance: float
    final_balance: float
    net_profit: float

    total_trades: int
    winning_trades: int
    losing_trades: int

    win_rate: float

    # ---- MetricsVector (Phase 1) ----
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    trades: Any = None
    equity_curve: Any = None


# ==========================================================
# ENGINE
# ==========================================================

class BacktestEngine:
    """
    Historical backtesting engine.

    Receives a DataFrame that has already passed through
    src.indicators.add_indicators().

    The engine itself is responsible only for:
        - validating prepared historical data
        - creating a clean TradeEngine
        - executing the historical run
        - extracting statistics
        - returning BacktestResult
    """

    def __init__(
        self,
        initial_balance: float | None = None,
        minimum_rows: int = MINIMUM_ROWS,
    ) -> None:

        if initial_balance is not None:

            if initial_balance <= 0:

                raise ValueError(
                    "initial_balance must be greater than zero."
                )

            self.initial_balance = float(
                initial_balance
            )

        else:

            self.initial_balance = None

        if type(minimum_rows) is not int:

            raise TypeError(
                "minimum_rows must be an integer."
            )

        if minimum_rows <= 0:

            raise ValueError(
                "minimum_rows must be greater than zero."
            )

        self.minimum_rows = minimum_rows

        # A TradeEngine instance is kept publicly available
        # because existing tests and project code use it.
        self.trade_engine = self._create_trade_engine()

        # Stores the latest result.
        self._result: BacktestResult | None = None


    # ======================================================
    # TRADE ENGINE FACTORY
    # ======================================================

    def _create_trade_engine(self) -> TradeEngine:
        """
        Create a completely fresh TradeEngine.

        This is important because TradeEngine contains
        mutable state:

            - balance
            - equity
            - positions
            - closed_positions
            - position_counter

        A new TradeEngine guarantees that every backtest
        starts from a clean state.
        """

        engine = TradeEngine()

        if self.initial_balance is not None:

            self._set_initial_balance(
                engine,
                self.initial_balance,
            )

        return engine

    # ======================================================
    # INTERNAL BALANCE SETTER
    # ======================================================

    @staticmethod
    def _set_initial_balance(
        engine: TradeEngine,
        balance: float,
    ) -> None:
        """
        Apply initial balance to the current TradeEngine.

        Supports the existing TradeEngine balance naming.
        """

        balance = float(balance)

        if hasattr(
            engine,
            "initial_balance",
        ):

            engine.initial_balance = balance

        if hasattr(
            engine,
            "balance",
        ):

            engine.balance = balance

        if hasattr(
            engine,
            "equity",
        ):

            engine.equity = balance

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

        if not isinstance(
            df,
            pd.DataFrame,
        ):

            raise TypeError(
                "BacktestEngine requires a pandas DataFrame."
            )

        if df.empty:

            raise ValueError(
                "Backtest data is empty."
            )

        missing = (
            REQUIRED_COLUMNS
            - set(df.columns)
        )

        if missing:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        if len(df) < self.minimum_rows:

            raise ValueError(
                f"Backtest requires at least "
                f"{self.minimum_rows} rows. "
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

        if (
            df[numeric_columns]
            .isna()
            .any()
            .any()
        ):

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
        """
        Safely extract the first available attribute
        from an object.
        """

        if obj is None:

            return default

        for name in names:

            if hasattr(
                obj,
                name,
            ):

                value = getattr(
                    obj,
                    name,
                )

                if value is not None:

                    return value

        return default

    # ======================================================
    # INITIAL BALANCE
    # ======================================================

    @staticmethod
    def _extract_initial_balance(
        engine: TradeEngine,
    ) -> float:
        """
        Extract the starting balance from TradeEngine.
        """

        value = BacktestEngine._get_value(
            engine,
            [
                "initial_balance",
                "starting_balance",
                "balance",
                "equity",
            ],
            0.0,
        )

        return float(value)

    # ======================================================
    # FINAL BALANCE
    # ======================================================

    @staticmethod
    def _extract_final_balance(
        engine: TradeEngine,
        engine_result: Any,
        initial_balance: float,
    ) -> float:
        """
        Extract final balance from TradeEngine result.
        """

        # Current TradeEngine.run() returns DataFrame.
        #
        # In that case the authoritative final balance is
        # stored on the TradeEngine itself.
        if isinstance(
            engine_result,
            pd.DataFrame,
        ):

            return float(
                getattr(
                    engine,
                    "balance",
                    initial_balance,
                )
            )

        # Support future object-based results as well.
        value = BacktestEngine._get_value(
            engine_result,
            [
                "final_balance",
                "balance",
                "equity",
                "current_balance",
            ],
            getattr(
                engine,
                "balance",
                initial_balance,
            ),
        )

        return float(value)

    # ======================================================
    # NET PROFIT
    # ======================================================

    @staticmethod
    def _extract_net_profit(
        engine: TradeEngine,
        engine_result: Any,
        initial_balance: float,
        final_balance: float,
    ) -> float:
        """
        Extract total net profit.

        For the current TradeEngine implementation,
        balance difference is used as the authoritative
        fallback.
        """

        if isinstance(
            engine_result,
            pd.DataFrame,
        ):

            if "net_profit" in engine_result.columns:

                return float(
                    engine_result[
                        "net_profit"
                    ].sum()
                )

            return float(
                final_balance
                - initial_balance
            )

        value = BacktestEngine._get_value(
            engine_result,
            [
                "net_profit",
                "profit",
                "total_profit",
            ],
            final_balance
            - initial_balance,
        )

        return float(value)

    # ======================================================
    # TRADE STATISTICS
    # ======================================================

    @staticmethod
    def _extract_trade_statistics(
        engine: TradeEngine,
        engine_result: Any,
    ) -> tuple[
        int,
        int,
        int,
        float,
    ]:
        """
        Extract:

            total_trades
            winning_trades
            losing_trades
            win_rate
        """

        total_trades = int(
            BacktestEngine._get_value(
                engine_result,
                [
                    "total_trades",
                    "trade_count",
                    "number_of_trades",
                ],
                BacktestEngine._get_value(
                    engine,
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
            BacktestEngine._get_value(
                engine_result,
                [
                    "winning_trades",
                    "wins",
                    "winning_count",
                ],
                BacktestEngine._get_value(
                    engine,
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
            BacktestEngine._get_value(
                engine_result,
                [
                    "losing_trades",
                    "losses",
                    "losing_count",
                ],
                BacktestEngine._get_value(
                    engine,
                    [
                        "losing_trades",
                        "losses",
                        "losing_count",
                    ],
                    0,
                ),
            )
        )

        # Current TradeEngine counts every non-winning
        # closed position as a loss.
        #
        # Keep the result internally consistent even if
        # a future engine implementation exposes slightly
        # different statistics.
        if total_trades > 0:

            if (
                winning_trades
                + losing_trades
                != total_trades
            ):

                losing_trades = (
                    total_trades
                    - winning_trades
                )

        else:

            winning_trades = 0
            losing_trades = 0

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

        return (
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
        )

    # ======================================================
    # TRADES EXTRACTION
    # ======================================================

    @staticmethod
    def _extract_trades(
        engine: TradeEngine,
        engine_result: Any,
    ) -> Any:
        """
        Extract completed trade history.
        """

        if isinstance(
            engine_result,
            pd.DataFrame,
        ):

            return engine_result.to_dict(
                orient="records"
            )

        trades = BacktestEngine._get_value(
            engine_result,
            [
                "trades",
                "trade_history",
                "history",
            ],
            BacktestEngine._get_value(
                engine,
                [
                    "trades",
                    "trade_history",
                    "history",
                ],
                [],
            ),
        )

        if trades is None:

            return []

        return trades

    # ======================================================
    # RISK METRICS (MetricsVector)
    # ======================================================

    @staticmethod
    def _infer_periods_per_year(
        timestamps: list,
    ) -> float:
        """
        Infer annualization factor from median bar interval.
        Falls back to 15m bars (35040/yr) when timestamps
        are missing or unparsable.
        """

        try:
            ts = pd.to_datetime(pd.Series(timestamps))
            dt = ts.diff().dropna()
            if len(dt) == 0:
                return 35040.0
            seconds = dt.dt.total_seconds().median()
            if seconds and seconds > 0:
                return 365.0 * 24.0 * 3600.0 / float(seconds)
        except Exception:
            pass
        return 35040.0

    @classmethod
    def _compute_risk_metrics(
        cls,
        trade_engine: TradeEngine,
        initial_balance: float,
    ) -> dict:
        """
        Compute the MetricsVector from closed trades and the
        per-bar equity curve recorded by TradeEngine.
        """

        import numpy as np

        closed = list(getattr(trade_engine, "closed_positions", []) or [])
        nets = [float(p.net_profit) for p in closed]

        wins = [n for n in nets if n > 0]
        losses = [n for n in nets if n < 0]

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        )

        expectancy = (
            sum(nets) / len(nets) if nets else 0.0
        )

        avg_win = (
            gross_profit / len(wins) if wins else 0.0
        )

        avg_loss = (
            -gross_loss / len(losses) if losses else 0.0
        )

        # ---- Equity curve ----
        curve = list(getattr(trade_engine, "equity_curve", []) or [])
        equity = [float(v) for _, v in curve]
        timestamps = [t for t, _ in curve]

        max_dd_pct = 0.0
        max_dd_abs = 0.0
        sharpe = 0.0
        sortino = 0.0

        if len(equity) >= 2:
            eq = np.asarray(equity, dtype=float)

            # Guard against account blow-up (eq <= 0): clip at tiny positive
            # so return math stays finite; bankruptcy is reported separately.
            eq = np.where(eq <= 0, 1e-9, eq)

            # Max drawdown on equity curve
            peaks = np.maximum.accumulate(eq)
            dd_abs = eq - peaks
            with np.errstate(divide="ignore", invalid="ignore"):
                dd_pct = np.where(peaks > 0, dd_abs / peaks, 0.0)
            min_i = int(np.argmin(dd_pct))
            max_dd_pct = float(dd_pct[min_i] * 100.0)
            max_dd_abs = float(dd_abs[min_i])

            # Per-bar returns
            rets = np.diff(eq) / eq[:-1]
            downside = rets[rets < 0]

            ppy = cls._infer_periods_per_year(timestamps)

            mean_r = float(rets.mean())
            std_r = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0

            if std_r > 0:
                sharpe = mean_r / std_r * (ppy ** 0.5)

            if len(downside) > 1:
                dstd = float(downside.std(ddof=1))
                if dstd > 0:
                    sortino = mean_r / dstd * (ppy ** 0.5)

        return {
            "max_drawdown_pct": round(max_dd_pct, 4),
            "max_drawdown_abs": round(max_dd_abs, 2),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "profit_factor": (
                round(profit_factor, 3)
                if profit_factor != float("inf")
                else float("inf")
            ),
            "expectancy": round(expectancy, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
        }

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

        Every call starts a completely fresh TradeEngine.
        """

        self.validate_data(df)

        # --------------------------------------------------
        # Prepare clean historical data
        # --------------------------------------------------

        data = df.copy()

        data = data.reset_index(
            drop=True
        )

        # --------------------------------------------------
        # IMPORTANT:
        # Create a fresh TradeEngine for every run.
        #
        # This prevents:
        #   - old balance
        #   - old positions
        #   - old closed trades
        #   - old position IDs
        #
        # from leaking into another backtest.
        # --------------------------------------------------

        self.trade_engine = (
            self._create_trade_engine()
        )

        # --------------------------------------------------
        # Initial balance
        # --------------------------------------------------

        initial_balance = (
            self._extract_initial_balance(
                self.trade_engine
            )
        )

        # --------------------------------------------------
        # Run Trade Engine
        # --------------------------------------------------

        engine_result = (
            self.trade_engine.run(
                data
            )
        )

        # --------------------------------------------------
        # Final balance
        # --------------------------------------------------

        final_balance = (
            self._extract_final_balance(
                self.trade_engine,
                engine_result,
                initial_balance,
            )
        )

        # --------------------------------------------------
        # Net profit
        # --------------------------------------------------

        net_profit = (
            self._extract_net_profit(
                self.trade_engine,
                engine_result,
                initial_balance,
                final_balance,
            )
        )

        # --------------------------------------------------
        # Trade statistics
        # --------------------------------------------------

        (
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
        ) = self._extract_trade_statistics(
            self.trade_engine,
            engine_result,
        )

        # --------------------------------------------------
        # Trades
        # --------------------------------------------------

        trades = self._extract_trades(
            self.trade_engine,
            engine_result,
        )

        # --------------------------------------------------
        # MetricsVector (risk-adjusted statistics)
        # --------------------------------------------------

        risk_metrics = self._compute_risk_metrics(
            self.trade_engine,
            initial_balance,
        )

        total_return_pct = (
            (final_balance - initial_balance)
            / initial_balance
            * 100.0
            if initial_balance
            else 0.0
        )

        # --------------------------------------------------
        # Build result
        # --------------------------------------------------

        result = BacktestResult(
            initial_balance=initial_balance,
            final_balance=final_balance,
            net_profit=net_profit,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return_pct=round(total_return_pct, 4),
            equity_curve=list(self.trade_engine.equity_curve),
            trades=trades,
            **risk_metrics,
        )

        self._result = result

        return result

    # ======================================================
    # RESULT
    # ======================================================

    @property
    def result(
        self,
    ) -> BacktestResult | None:
        """
        Return the latest backtest result.

        Returns None before the first run.
        """

        return self._result

    # ======================================================
    # REPORT
    # ======================================================

    @staticmethod
    def print_report(
        result: BacktestResult,
    ) -> None:
        """
        Print a human-readable backtest report.
        """

        if not isinstance(
            result,
            BacktestResult,
        ):

            raise TypeError(
                "result must be a BacktestResult."
            )

        print()

        print(
            "=" * 60
        )

        print(
            "QUANTAI BACKTEST REPORT"
        )

        print(
            "=" * 60
        )

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

        print(
            "-" * 60
        )

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

        print(
            "-" * 60
        )

        print(
            f"Total Return    : "
            f"{result.total_return_pct:.2f}%"
        )

        print(
            f"Max Drawdown    : "
            f"{result.max_drawdown_pct:.2f}%"
        )

        print(
            f"Profit Factor   : "
            f"{result.profit_factor}"
        )

        print(
            f"Sharpe (ann.)   : "
            f"{result.sharpe}"
        )

        print(
            f"Sortino (ann.)  : "
            f"{result.sortino}"
        )

        print(
            f"Expectancy/trade: "
            f"{result.expectancy}"
        )

        print(
            f"Avg Win / Loss  : "
            f"{result.avg_win} / {result.avg_loss}"
        )

        print(
            "=" * 60
        )

    # ======================================================
    # CONVENIENCE FUNCTION
    # ======================================================

def run_backtest(
    df: pd.DataFrame,
) -> BacktestResult:
    """
    Convenience wrapper.
    """

    engine = BacktestEngine()

    result = engine.run(
        df
    )

    engine.print_report(
        result
    )

    return result


# ==========================================================
# EXPORTS
# ==========================================================

__all__ = [
    "MINIMUM_ROWS",
    "REQUIRED_COLUMNS",
    "BacktestResult",
    "BacktestEngine",
    "run_backtest",
]
