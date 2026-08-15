"""
=========================================================
QuantAI Professional v5
Performance Analyzer
=========================================================

Analyzes completed historical trades produced by:

    BacktestEngine
        ↓
    TradeEngine
        ↓
    PerformanceAnalyzer

Responsibilities:
    - validate completed trade history
    - calculate performance statistics
    - calculate profitability metrics
    - calculate drawdown metrics
    - calculate trade statistics
    - preserve source DataFrame
    - provide a persistent PerformanceResult
    - print a human-readable performance report

The Performance Analyzer does NOT:
    - load market data
    - use CCXT
    - calculate indicators
    - generate strategy signals
    - open or close positions
    - modify TradeEngine
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

REQUIRED_COLUMNS = {
    "net_profit",
}


# =========================================================
# RESULT
# =========================================================

@dataclass
class PerformanceResult:
    """
    Final performance statistics.
    """

    initial_balance: float
    final_balance: float
    net_profit: float
    net_profit_percent: float

    total_trades: int
    winning_trades: int
    losing_trades: int

    win_rate: float

    average_profit: float
    profit_factor: float

    gross_profit: float
    gross_loss: float

    largest_win: float
    largest_loss: float

    average_win: float
    average_loss: float

    max_drawdown: float
    max_drawdown_percent: float

    average_trade: float

    trades: Any = None


# =========================================================
# ANALYZER
# =========================================================

class PerformanceAnalyzer:
    """
    Analyze completed trades.

    The analyzer expects a DataFrame produced by
    TradeEngine.to_dataframe() or BacktestEngine.

    Required minimum column:

        net_profit

    Optional columns are used when available:

        balance
        gross_profit
        commission
        close_reason
        side
        entry_time
        exit_time
        bars
        confidence
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
    ) -> None:

        if initial_balance <= 0:

            raise ValueError(
                "initial_balance must be greater than zero."
            )

        self.initial_balance = float(
            initial_balance
        )

        self._result: PerformanceResult | None = None

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate_data(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate trade history.

        Empty DataFrames are accepted and represent a
        legitimate zero-trade performance case.
        """

        if not isinstance(
            df,
            pd.DataFrame,
        ):

            raise TypeError(
                "PerformanceAnalyzer requires a pandas DataFrame."
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

        # -------------------------------------------------
        # Empty trade history is valid.
        # -------------------------------------------------

        if df.empty:

            return

        # -------------------------------------------------
        # Required numeric column.
        # -------------------------------------------------

        if not pd.api.types.is_numeric_dtype(
            df["net_profit"]
        ):

            raise TypeError(
                "Column 'net_profit' must be numeric."
            )

        # -------------------------------------------------
        # NaN protection.
        # -------------------------------------------------

        if df["net_profit"].isna().any():

            raise ValueError(
                "Trade data contains NaN values "
                "in 'net_profit'."
            )

        # -------------------------------------------------
        # Optional numeric columns.
        # -------------------------------------------------

        optional_numeric_columns = [
            "gross_profit",
            "commission",
            "balance",
            "quantity",
            "confidence",
            "bars",
        ]

        for column in optional_numeric_columns:

            if column not in df.columns:

                continue

            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):

                raise TypeError(
                    f"Column '{column}' must be numeric."
                )

            if df[column].isna().any():

                raise ValueError(
                    f"Trade data contains NaN values "
                    f"in '{column}'."
                )

    # =====================================================
    # VALIDATE TRADES
    # =====================================================

    def validate_trades(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Backward-compatible validation method.

        Kept as an alias for existing project code.
        """

        self.validate_data(df)

    # =====================================================
    # INTERNAL HELPERS
    # =====================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value to float.
        """

        if value is None:

            return default

        try:

            result = float(value)

        except (
            TypeError,
            ValueError,
        ):

            return default

        if pd.isna(result):

            return default

        return result

    # =====================================================
    # FINAL BALANCE
    # =====================================================

    def _calculate_final_balance(
        self,
        df: pd.DataFrame,
    ) -> float:
        """
        Calculate final balance.

        If a balance column exists, its last valid value
        is authoritative.

        Otherwise:

            final_balance =
                initial_balance + total_net_profit
        """

        if df.empty:

            return self.initial_balance

        if "balance" in df.columns:

            balances = pd.to_numeric(
                df["balance"],
                errors="coerce",
            )

            balances = balances.dropna()

            if not balances.empty:

                return float(
                    balances.iloc[-1]
                )

        total_profit = float(
            pd.to_numeric(
                df["net_profit"],
                errors="coerce",
            ).sum()
        )

        return (
            self.initial_balance
            + total_profit
        )

    # =====================================================
    # GROSS PROFIT / LOSS
    # =====================================================

    @staticmethod
    def _calculate_gross_profit(
        df: pd.DataFrame,
    ) -> float:
        """
        Calculate total positive net-profit amount.

        Profit factor is based on positive and negative
        net profits because commissions are already included
        in TradeEngine net_profit.
        """

        if df.empty:

            return 0.0

        profits = pd.to_numeric(
            df["net_profit"],
            errors="coerce",
        ).fillna(0.0)

        return float(
            profits[profits > 0].sum()
        )

    @staticmethod
    def _calculate_gross_loss(
        df: pd.DataFrame,
    ) -> float:
        """
        Calculate absolute total losses.
        """

        if df.empty:

            return 0.0

        profits = pd.to_numeric(
            df["net_profit"],
            errors="coerce",
        ).fillna(0.0)

        return float(
            abs(
                profits[profits < 0].sum()
            )
        )

    # =====================================================
    # PROFIT FACTOR
    # =====================================================

    @staticmethod
    def _calculate_profit_factor(
        gross_profit: float,
        gross_loss: float,
    ) -> float:
        """
        Profit Factor:

            gross profits / absolute gross losses

        If there are no losses:

            - 0 trades / 0 profit -> 0.0
            - profitable trades only -> infinity
        """

        if gross_loss == 0.0:

            if gross_profit > 0.0:

                return float("inf")

            return 0.0

        return (
            gross_profit
            / gross_loss
        )

    # =====================================================
    # WIN / LOSS STATISTICS
    # =====================================================

    @staticmethod
    def _calculate_trade_statistics(
        df: pd.DataFrame,
    ) -> tuple[
        int,
        int,
        int,
        float,
        float,
        float,
        float,
        float,
    ]:
        """
        Calculate:

            total_trades
            winning_trades
            losing_trades
            win_rate
            average_profit
            average_win
            average_loss
            average_trade
        """

        if df.empty:

            return (
                0,
                0,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

        profits = pd.to_numeric(
            df["net_profit"],
            errors="coerce",
        ).fillna(0.0)

        total_trades = int(
            len(profits)
        )

        winning = profits[
            profits > 0
        ]

        losing = profits[
            profits <= 0
        ]

        winning_trades = int(
            len(winning)
        )

        losing_trades = int(
            len(losing)
        )

        if total_trades > 0:

            win_rate = (
                winning_trades
                / total_trades
                * 100.0
            )

        else:

            win_rate = 0.0

        if total_trades > 0:

            average_profit = float(
                profits.sum()
                / total_trades
            )

        else:

            average_profit = 0.0

        if winning_trades > 0:

            average_win = float(
                winning.mean()
            )

        else:

            average_win = 0.0

        if losing_trades > 0:

            average_loss = float(
                losing.mean()
            )

        else:

            average_loss = 0.0

        average_trade = average_profit

        return (
            total_trades,
            winning_trades,
            losing_trades,
            float(win_rate),
            average_profit,
            average_win,
            average_loss,
            average_trade,
        )

    # =====================================================
    # EXTREME TRADES
    # =====================================================

    @staticmethod
    def _calculate_largest_win(
        df: pd.DataFrame,
    ) -> float:
        """
        Return largest winning trade.
        """

        if df.empty:

            return 0.0

        profits = pd.to_numeric(
            df["net_profit"],
            errors="coerce",
        ).fillna(0.0)

        winning = profits[
            profits > 0
        ]

        if winning.empty:

            return 0.0

        return float(
            winning.max()
        )

    @staticmethod
    def _calculate_largest_loss(
        df: pd.DataFrame,
    ) -> float:
        """
        Return largest losing trade.

        The value remains negative because this represents
        the actual worst trade result.
        """

        if df.empty:

            return 0.0

        profits = pd.to_numeric(
            df["net_profit"],
            errors="coerce",
        ).fillna(0.0)

        losing = profits[
            profits < 0
        ]

        if losing.empty:

            return 0.0

        return float(
            losing.min()
        )

    # =====================================================
    # EQUITY CURVE
    # =====================================================

    def _calculate_equity_curve(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Build equity curve from completed trade profits.

        When a balance column exists, that column is used
        directly.

        Compatible with pandas 3.x.
        """

        if df.empty:

            return pd.Series(
                dtype=float
            )

        if "balance" in df.columns:

            balances = pd.to_numeric(
                df["balance"],
                errors="coerce",
            )

            if not balances.isna().all():

                # pandas 3.x compatible:
                # fillna(method="ffill") is no longer supported.
                return balances.ffill()

        profits = pd.to_numeric(
            df["net_profit"],
            errors="coerce",
        ).fillna(0.0)

        return (
            self.initial_balance
            + profits.cumsum()
        )

    # =====================================================
    # MAX DRAWDOWN
    # =====================================================

    def _calculate_drawdown(
        self,
        df: pd.DataFrame,
    ) -> tuple[
        float,
        float,
    ]:
        """
        Calculate maximum drawdown.

        Public result convention:

            max_drawdown >= 0
            max_drawdown_percent >= 0

        Internally the drawdown is calculated as:

            peak - equity

        Therefore a decline in equity produces a positive
        drawdown value.

        Example:

            equity:
                1000
                1005
                1002
                1007

            peak:
                1000
                1005
                1005
                1007

            drawdown:
                0
                0
                3
                0

            max_drawdown = 3
        """

        equity = self._calculate_equity_curve(
            df
        )

        if equity.empty:

            return (
                0.0,
                0.0,
            )

        # -------------------------------------------------
        # Running equity peak.
        # -------------------------------------------------

        running_peak = equity.cummax()

        # -------------------------------------------------
        # Positive drawdown amount.
        #
        # IMPORTANT:
        # Do NOT calculate equity - peak here because the
        # public API expects drawdown to be non-negative.
        # -------------------------------------------------

        drawdown = (
            running_peak
            - equity
        )

        # Numerical safety.
        drawdown = drawdown.clip(
            lower=0.0
        )

        max_drawdown = float(
            drawdown.max()
        )

        # -------------------------------------------------
        # Drawdown percentage.
        # -------------------------------------------------

        denominator = running_peak.replace(
            0,
            pd.NA,
        )

        drawdown_percent = (
            drawdown
            / denominator
            * 100.0
        )

        drawdown_percent = (
            drawdown_percent
            .fillna(0.0)
        )

        drawdown_percent = drawdown_percent.clip(
            lower=0.0
        )

        max_drawdown_percent = float(
            drawdown_percent.max()
        )

        return (
            max_drawdown,
            max_drawdown_percent,
        )

    # =====================================================
    # TRADE EXTRACTION
    # =====================================================

    @staticmethod
    def _extract_trades(
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Convert trade DataFrame to independent records.
        """

        if df.empty:

            return []

        return df.to_dict(
            orient="records"
        )

    # =====================================================
    # ANALYZE
    # =====================================================

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> PerformanceResult:
        """
        Analyze completed trade history.
        """

        self.validate_data(df)

        # -------------------------------------------------
        # Protect original DataFrame.
        # -------------------------------------------------

        data = df.copy(
            deep=True
        )

        # -------------------------------------------------
        # Empty result.
        # -------------------------------------------------

        if data.empty:

            result = PerformanceResult(
                initial_balance=self.initial_balance,
                final_balance=self.initial_balance,
                net_profit=0.0,
                net_profit_percent=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                average_profit=0.0,
                profit_factor=0.0,
                gross_profit=0.0,
                gross_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                average_win=0.0,
                average_loss=0.0,
                max_drawdown=0.0,
                max_drawdown_percent=0.0,
                average_trade=0.0,
                trades=[],
            )

            self._result = result

            return result

        # -------------------------------------------------
        # Final balance.
        # -------------------------------------------------

        final_balance = (
            self._calculate_final_balance(
                data
            )
        )

        # -------------------------------------------------
        # Net profit.
        # -------------------------------------------------

        net_profit = float(
            pd.to_numeric(
                data["net_profit"],
                errors="coerce",
            ).sum()
        )

        # -------------------------------------------------
        # Profit percentage.
        # -------------------------------------------------

        if self.initial_balance != 0:

            net_profit_percent = (
                net_profit
                / self.initial_balance
                * 100.0
            )

        else:

            net_profit_percent = 0.0

        # -------------------------------------------------
        # Trade statistics.
        # -------------------------------------------------

        (
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            average_profit,
            average_win,
            average_loss,
            average_trade,
        ) = self._calculate_trade_statistics(
            data
        )

        # -------------------------------------------------
        # Gross profit / loss.
        # -------------------------------------------------

        gross_profit = (
            self._calculate_gross_profit(
                data
            )
        )

        gross_loss = (
            self._calculate_gross_loss(
                data
            )
        )

        # -------------------------------------------------
        # Profit factor.
        # -------------------------------------------------

        profit_factor = (
            self._calculate_profit_factor(
                gross_profit,
                gross_loss,
            )
        )

        # -------------------------------------------------
        # Largest trades.
        # -------------------------------------------------

        largest_win = (
            self._calculate_largest_win(
                data
            )
        )

        largest_loss = (
            self._calculate_largest_loss(
                data
            )
        )

        # -------------------------------------------------
        # Drawdown.
        # -------------------------------------------------

        (
            max_drawdown,
            max_drawdown_percent,
        ) = self._calculate_drawdown(
            data
        )

        # -------------------------------------------------
        # Trade records.
        # -------------------------------------------------

        trades = self._extract_trades(
            data
        )

        # -------------------------------------------------
        # Result.
        # -------------------------------------------------

        result = PerformanceResult(
            initial_balance=self.initial_balance,
            final_balance=float(final_balance),
            net_profit=float(net_profit),
            net_profit_percent=float(
                net_profit_percent
            ),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=float(win_rate),
            average_profit=float(
                average_profit
            ),
            profit_factor=float(
                profit_factor
            ),
            gross_profit=float(
                gross_profit
            ),
            gross_loss=float(
                gross_loss
            ),
            largest_win=float(
                largest_win
            ),
            largest_loss=float(
                largest_loss
            ),
            average_win=float(
                average_win
            ),
            average_loss=float(
                average_loss
            ),
            max_drawdown=float(
                max_drawdown
            ),
            max_drawdown_percent=float(
                max_drawdown_percent
            ),
            average_trade=float(
                average_trade
            ),
            trades=trades,
        )

        self._result = result

        return result

    # =====================================================
    # RESULT PROPERTY
    # =====================================================

    @property
    def result(
        self,
    ) -> PerformanceResult | None:
        """
        Return the latest analysis result.

        Returns None before the first successful analysis.
        """

        return self._result

    # =====================================================
    # REPORT
    # =====================================================

    @staticmethod
    def print_report(
        result: PerformanceResult,
    ) -> None:
        """
        Print a human-readable performance report.
        """

        if not isinstance(
            result,
            PerformanceResult,
        ):

            raise TypeError(
                "result must be a PerformanceResult."
            )

        print()

        print(
            "=" * 65
        )

        print(
            "QUANTAI PERFORMANCE REPORT"
        )

        print(
            "=" * 65
        )

        print(
            f"Initial Balance       : "
            f"{result.initial_balance:.2f}"
        )

        print(
            f"Final Balance         : "
            f"{result.final_balance:.2f}"
        )

        print(
            f"Net Profit            : "
            f"{result.net_profit:.2f}"
        )

        print(
            f"Net Profit %          : "
            f"{result.net_profit_percent:.2f}%"
        )

        print(
            "-" * 65
        )

        print(
            f"Total Trades          : "
            f"{result.total_trades}"
        )

        print(
            f"Winning Trades        : "
            f"{result.winning_trades}"
        )

        print(
            f"Losing Trades         : "
            f"{result.losing_trades}"
        )

        print(
            f"Win Rate              : "
            f"{result.win_rate:.2f}%"
        )

        print(
            f"Average Profit        : "
            f"{result.average_profit:.6f}"
        )

        print(
            f"Average Win           : "
            f"{result.average_win:.6f}"
        )

        print(
            f"Average Loss          : "
            f"{result.average_loss:.6f}"
        )

        print(
            "-" * 65
        )

        print(
            f"Gross Profit          : "
            f"{result.gross_profit:.6f}"
        )

        print(
            f"Gross Loss            : "
            f"{result.gross_loss:.6f}"
        )

        if result.profit_factor == float("inf"):

            profit_factor_text = "inf"

        else:

            profit_factor_text = (
                f"{result.profit_factor:.6f}"
            )

        print(
            f"Profit Factor         : "
            f"{profit_factor_text}"
        )

        print(
            f"Largest Win           : "
            f"{result.largest_win:.6f}"
        )

        print(
            f"Largest Loss          : "
            f"{result.largest_loss:.6f}"
        )

        print(
            "-" * 65
        )

        # -------------------------------------------------
        # IMPORTANT:
        # Tests expect the exact phrase
        # "Maximum Drawdown".
        # -------------------------------------------------

        print(
            f"Maximum Drawdown      : "
            f"{result.max_drawdown:.6f}"
        )

        print(
            f"Maximum Drawdown %    : "
            f"{result.max_drawdown_percent:.6f}%"
        )

        print(
            "=" * 65
        )

    # =====================================================
    # CONVENIENCE
    # =====================================================

    def analyze_and_report(
        self,
        df: pd.DataFrame,
    ) -> PerformanceResult:
        """
        Analyze trades and immediately print the report.
        """

        result = self.analyze(
            df
        )

        self.print_report(
            result
        )

        return result


# =========================================================
# PUBLIC RUNNER
# =========================================================

def analyze_performance(
    trades: pd.DataFrame,
    initial_balance: float = 1000.0,
) -> PerformanceResult:
    """
    Convenience function for performance analysis.
    """

    analyzer = PerformanceAnalyzer(
        initial_balance=initial_balance,
    )

    result = analyzer.analyze(
        trades
    )

    analyzer.print_report(
        result
    )

    return result


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "REQUIRED_COLUMNS",
    "PerformanceResult",
    "PerformanceAnalyzer",
    "analyze_performance",
]
