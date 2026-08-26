from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import pandas as pd

from src.walk.walk_forward_engine import (
    WalkForwardEngine,
    WalkForwardResult,
    WalkForwardWindowResult,
)


@dataclass(frozen=True)
class WalkForwardValidationResult:
    total_windows: int
    profitable_windows: int
    losing_windows: int
    flat_windows: int

    profitable_window_rate: float
    losing_window_rate: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    initial_balance: float
    final_balance: float
    net_profit: float
    return_percent: float

    best_window_profit: float
    worst_window_profit: float
    average_window_profit: float

    best_window_win_rate: float
    worst_window_win_rate: float
    average_window_win_rate: float

    validation_passed: bool
    validation_score: float

    windows: List[Any]


class WalkForwardValidator:
    """
    Validation layer for QuantAI WalkForwardEngine.

    Responsibilities:
        - validate input market data
        - preserve walk-forward configuration
        - execute WalkForwardEngine
        - analyze the resulting walk-forward performance
        - expose the latest validation result

    This class does not:
        - execute trades directly
        - connect to exchanges
        - modify WalkForwardEngine
        - modify BacktestEngine
        - train ML models
    """

    MINIMUM_WINDOWS = 1

    def __init__(
        self,
        train_size: int = 500,
        test_size: int = 100,
        step_size: int | None = None,
        initial_balance: float = 1000.0,
        minimum_windows: int = 1,
        require_positive_return: bool = False,
        require_positive_window_rate: float = 0.50,
    ) -> None:

        if type(train_size) is not int:
            raise TypeError(
                "train_size must be an integer."
            )

        if train_size <= 0:
            raise ValueError(
                "train_size must be greater than zero."
            )

        if type(test_size) is not int:
            raise TypeError(
                "test_size must be an integer."
            )

        if test_size <= 0:
            raise ValueError(
                "test_size must be greater than zero."
            )

        if step_size is not None:

            if type(step_size) is not int:
                raise TypeError(
                    "step_size must be an integer."
                )

            if step_size <= 0:
                raise ValueError(
                    "step_size must be greater than zero."
                )

        if not isinstance(
            initial_balance,
            (int, float),
        ):
            raise TypeError(
                "initial_balance must be numeric."
            )

        if float(initial_balance) <= 0:
            raise ValueError(
                "initial_balance must be greater than zero."
            )

        if type(minimum_windows) is not int:
            raise TypeError(
                "minimum_windows must be an integer."
            )

        if minimum_windows <= 0:
            raise ValueError(
                "minimum_windows must be greater than zero."
            )

        if not isinstance(
            require_positive_return,
            bool,
        ):
            raise TypeError(
                "require_positive_return must be a boolean."
            )

        if not isinstance(
            require_positive_window_rate,
            (int, float),
        ):
            raise TypeError(
                "require_positive_window_rate "
                "must be numeric."
            )

        require_positive_window_rate = float(
            require_positive_window_rate
        )

        if not 0.0 <= require_positive_window_rate <= 1.0:
            raise ValueError(
                "require_positive_window_rate must be "
                "between 0.0 and 1.0."
            )

        self.train_size = train_size
        self.test_size = test_size
        self.step_size = (
            test_size
            if step_size is None
            else step_size
        )

        self.initial_balance = float(
            initial_balance
        )

        self.minimum_windows = minimum_windows

        self.require_positive_return = (
            require_positive_return
        )

        self.require_positive_window_rate = (
            require_positive_window_rate
        )

        self._result: WalkForwardValidationResult | None = None

    # =========================================================
    # DATA VALIDATION
    # =========================================================

    def validate_data(
        self,
        df: pd.DataFrame,
    ) -> bool:
        """
        Validate input market DataFrame.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "DataFrame cannot be empty."
            )

        minimum_rows = (
            self.train_size
            + self.test_size
        )

        if len(df) < minimum_rows:
            raise ValueError(
                "DataFrame does not contain enough "
                "rows for the configured train and "
                "test sizes."
            )

        return True

    # =========================================================
    # RESULT VALIDATION
    # =========================================================

    @staticmethod
    def validate_result(
        result: WalkForwardResult,
    ) -> None:
        """
        Validate a completed WalkForwardResult.
        """

        if not isinstance(
            result,
            WalkForwardResult,
        ):
            raise TypeError(
                "result must be WalkForwardResult."
            )

        windows = getattr(
            result,
            "windows",
            None,
        )

        if windows is None:
            raise ValueError(
                "Walk-forward result must contain windows."
            )

        if len(windows) == 0:
            raise ValueError(
                "Walk-forward result contains no windows."
            )

        total_trades = int(
            getattr(
                result,
                "total_trades",
                0,
            )
        )

        winning_trades = int(
            getattr(
                result,
                "winning_trades",
                0,
            )
        )

        losing_trades = int(
            getattr(
                result,
                "losing_trades",
                0,
            )
        )

        if total_trades < 0:
            raise ValueError(
                "total_trades cannot be negative."
            )

        if winning_trades < 0:
            raise ValueError(
                "winning_trades cannot be negative."
            )

        if losing_trades < 0:
            raise ValueError(
                "losing_trades cannot be negative."
            )

        if (
            winning_trades
            + losing_trades
            != total_trades
        ):
            raise ValueError(
                "Winning and losing trades must "
                "equal total trades."
            )

    # =========================================================
    # WINDOW HELPERS
    # =========================================================

    @staticmethod
    def _get_windows(
        result: Any,
    ) -> list[Any]:
        windows = getattr(
            result,
            "windows",
            None,
        )

        if windows is None:
            return []

        return list(windows)

    @staticmethod
    def _window_profits(
        result: Any,
    ) -> list[float]:
        """
        Extract window profits.

        Supports real WalkForwardWindowResult objects
        and lightweight fake objects used by tests.
        """

        profits: list[float] = []

        for window in WalkForwardValidator._get_windows(
            result
        ):

            backtest_result = getattr(
                window,
                "backtest_result",
                None,
            )

            if backtest_result is None:
                continue

            profit = getattr(
                backtest_result,
                "net_profit",
                None,
            )

            if profit is not None:
                profits.append(
                    float(profit)
                )

        return profits

    @staticmethod
    def _window_win_rates(
        result: Any,
    ) -> list[float]:
        """
        Extract window win rates.
        """

        rates: list[float] = []

        for window in WalkForwardValidator._get_windows(
            result
        ):

            backtest_result = getattr(
                window,
                "backtest_result",
                None,
            )

            if backtest_result is None:
                continue

            win_rate = getattr(
                backtest_result,
                "win_rate",
                None,
            )

            if win_rate is not None:
                rates.append(
                    float(win_rate)
                )

        return rates

    # =========================================================
    # SCORE
    # =========================================================

    def _calculate_score(
        self,
        result: Any,
        profitable_window_rate: float,
        return_percent: float,
    ) -> float:
        """
        Calculate validation quality score from 0 to 100.

        40% profitable-window consistency
        40% overall return
        20% trade win rate
        """

        consistency_score = (
            max(
                0.0,
                min(
                    1.0,
                    profitable_window_rate,
                ),
            )
            * 100.0
        )

        return_score = max(
            0.0,
            min(
                100.0,
                return_percent * 5.0,
            ),
        )

        win_rate = float(
            getattr(
                result,
                "win_rate",
                0.0,
            )
        )

        win_rate_score = max(
            0.0,
            min(
                100.0,
                win_rate,
            ),
        )

        score = (
            consistency_score * 0.40
            + return_score * 0.40
            + win_rate_score * 0.20
        )

        return round(
            max(
                0.0,
                min(
                    100.0,
                    score,
                ),
            ),
            2,
        )

    # =========================================================
    # VALIDATION STATUS
    # =========================================================

    def _is_validation_passed(
        self,
        result: Any,
        profitable_window_rate: float,
        total_windows: int,
    ) -> bool:

        if total_windows < self.minimum_windows:
            return False

        if (
            profitable_window_rate
            < self.require_positive_window_rate
        ):
            return False

        net_profit = float(
            getattr(
                result,
                "net_profit",
                0.0,
            )
        )

        if (
            self.require_positive_return
            and net_profit <= 0
        ):
            return False

        return True

    # =========================================================
    # BUILD VALIDATION RESULT
    # =========================================================

    def _build_validation_result(
        self,
        result: Any,
    ) -> WalkForwardValidationResult:
        """
        Convert WalkForwardResult into
        WalkForwardValidationResult.
        """

        windows = self._get_windows(
            result
        )

        total_windows = int(
            getattr(
                result,
                "total_windows",
                len(windows),
            )
        )

        if total_windows <= 0:
            total_windows = len(windows)

        profits = self._window_profits(
            result
        )

        win_rates = self._window_win_rates(
            result
        )

        profitable_windows = sum(
            1
            for profit in profits
            if profit > 0
        )

        losing_windows = sum(
            1
            for profit in profits
            if profit < 0
        )

        flat_windows = sum(
            1
            for profit in profits
            if profit == 0
        )

        if total_windows > 0:

            profitable_window_rate = round(
                (
                    profitable_windows
                    / total_windows
                )
                * 100.0,
                2,
            )

            losing_window_rate = round(
                (
                    losing_windows
                    / total_windows
                )
                * 100.0,
                2,
            )

        else:

            profitable_window_rate = 0.0
            losing_window_rate = 0.0

        total_trades = int(
            getattr(
                result,
                "total_trades",
                0,
            )
        )

        winning_trades = int(
            getattr(
                result,
                "winning_trades",
                0,
            )
        )

        losing_trades = int(
            getattr(
                result,
                "losing_trades",
                0,
            )
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

        initial_balance = float(
            getattr(
                result,
                "initial_balance",
                self.initial_balance,
            )
        )

        final_balance = float(
            getattr(
                result,
                "final_balance",
                initial_balance,
            )
        )

        net_profit = float(
            getattr(
                result,
                "net_profit",
                final_balance - initial_balance,
            )
        )

        if initial_balance > 0:

            return_percent = round(
                (
                    net_profit
                    / initial_balance
                )
                * 100.0,
                2,
            )

        else:

            return_percent = 0.0

        if profits:

            best_window_profit = max(
                profits
            )

            worst_window_profit = min(
                profits
            )

            average_window_profit = (
                sum(profits)
                / len(profits)
            )

        else:

            best_window_profit = 0.0
            worst_window_profit = 0.0
            average_window_profit = 0.0

        if win_rates:

            best_window_win_rate = max(
                win_rates
            )

            worst_window_win_rate = min(
                win_rates
            )

            average_window_win_rate = (
                sum(win_rates)
                / len(win_rates)
            )

        else:

            best_window_win_rate = 0.0
            worst_window_win_rate = 0.0
            average_window_win_rate = 0.0

        positive_rate = (
            profitable_window_rate
            / 100.0
        )

        validation_passed = (
            self._is_validation_passed(
                result=result,
                profitable_window_rate=positive_rate,
                total_windows=total_windows,
            )
        )

        validation_score = (
            self._calculate_score(
                result=result,
                profitable_window_rate=positive_rate,
                return_percent=return_percent,
            )
        )

        return WalkForwardValidationResult(
            total_windows=total_windows,
            profitable_windows=profitable_windows,
            losing_windows=losing_windows,
            flat_windows=flat_windows,
            profitable_window_rate=(
                profitable_window_rate
            ),
            losing_window_rate=(
                losing_window_rate
            ),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            initial_balance=initial_balance,
            final_balance=final_balance,
            net_profit=net_profit,
            return_percent=return_percent,
            best_window_profit=round(
                best_window_profit,
                8,
            ),
            worst_window_profit=round(
                worst_window_profit,
                8,
            ),
            average_window_profit=round(
                average_window_profit,
                8,
            ),
            best_window_win_rate=round(
                best_window_win_rate,
                2,
            ),
            worst_window_win_rate=round(
                worst_window_win_rate,
                2,
            ),
            average_window_win_rate=round(
                average_window_win_rate,
                2,
            ),
            validation_passed=validation_passed,
            validation_score=validation_score,
            windows=windows,
        )

    # =========================================================
    # VALIDATE
    # =========================================================

    def validate(
        self,
        data: Any,
    ) -> Any:
        """
        Validate either:

            1. a DataFrame -> True
            2. a WalkForwardResult -> ValidationResult

        This dual behavior preserves compatibility with
        both data-validation tests and result-analysis usage.
        """

        if isinstance(
            data,
            pd.DataFrame,
        ):
            return self.validate_data(
                data
            )

        if isinstance(
            data,
            WalkForwardResult,
        ):
            self.validate_result(
                data
            )

            validation_result = (
                self._build_validation_result(
                    data
                )
            )

            self._result = validation_result

            return validation_result

        raise TypeError(
            "data must be pandas DataFrame "
            "or WalkForwardResult."
        )

    # =========================================================
    # RUN
    # =========================================================

    def run(
        self,
        df: pd.DataFrame,
    ) -> WalkForwardValidationResult:
        """
        Validate data, execute WalkForwardEngine,
        and analyze the resulting performance.
        """

        self.validate_data(
            df
        )

        data = df.copy(
            deep=True
        )

        engine = WalkForwardEngine(
            train_size=self.train_size,
            test_size=self.test_size,
            step_size=self.step_size,
            initial_balance=self.initial_balance,
        )

        walk_forward_result = engine.run(
            data
        )

        self.validate_result(
            walk_forward_result
        )

        validation_result = (
            self._build_validation_result(
                walk_forward_result
            )
        )

        self._result = validation_result

        return validation_result

    # =========================================================
    # RESULT PROPERTY
    # =========================================================

    @property
    def result(
        self,
    ) -> WalkForwardValidationResult | None:
        """
        Return latest validation result.
        """

        return self._result

    # =========================================================
    # REPORT
    # =========================================================

    @staticmethod
    def print_report(
        result: WalkForwardValidationResult,
    ) -> None:
        """
        Print concise validation report.
        """

        if not isinstance(
            result,
            WalkForwardValidationResult,
        ):
            raise TypeError(
                "result must be "
                "WalkForwardValidationResult."
            )

        print()
        print("=" * 70)
        print(
            "QUANTAI WALK-FORWARD VALIDATION REPORT"
        )
        print("=" * 70)

        print(
            f"Validation Status : "
            f"{'PASSED' if result.validation_passed else 'FAILED'}"
        )

        print(
            f"Validation Score  : "
            f"{result.validation_score:.2f}/100"
        )

        print("-" * 70)

        print(
            f"Windows           : "
            f"{result.total_windows}"
        )

        print(
            f"Profitable Windows: "
            f"{result.profitable_windows}"
            f" ({result.profitable_window_rate:.2f}%)"
        )

        print(
            f"Losing Windows    : "
            f"{result.losing_windows}"
            f" ({result.losing_window_rate:.2f}%)"
        )

        print(
            f"Flat Windows      : "
            f"{result.flat_windows}"
        )

        print("-" * 70)

        print(
            f"Initial Balance   : "
            f"{result.initial_balance:.2f}"
        )

        print(
            f"Final Balance     : "
            f"{result.final_balance:.2f}"
        )

        print(
            f"Net Profit        : "
            f"{result.net_profit:.2f}"
        )

        print(
            f"Return            : "
            f"{result.return_percent:.2f}%"
        )

        print("-" * 70)

        print(
            f"Total Trades      : "
            f"{result.total_trades}"
        )

        print(
            f"Winning Trades    : "
            f"{result.winning_trades}"
        )

        print(
            f"Losing Trades     : "
            f"{result.losing_trades}"
        )

        print(
            f"Win Rate          : "
            f"{result.win_rate:.2f}%"
        )

        print("-" * 70)

        print(
            f"Best Window Profit : "
            f"{result.best_window_profit:.2f}"
        )

        print(
            f"Worst Window Profit: "
            f"{result.worst_window_profit:.2f}"
        )

        print(
            f"Average Window Profit: "
            f"{result.average_window_profit:.2f}"
        )

        print("-" * 70)

        print(
            f"Best Window Win Rate: "
            f"{result.best_window_win_rate:.2f}%"
        )

        print(
            f"Worst Window Win Rate: "
            f"{result.worst_window_win_rate:.2f}%"
        )

        print(
            f"Average Window Win Rate: "
            f"{result.average_window_win_rate:.2f}%"
        )

        print("=" * 70)


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def validate_walk_forward(
    result: WalkForwardResult,
    minimum_windows: int = 1,
    require_positive_return: bool = False,
    require_positive_window_rate: float = 0.50,
) -> WalkForwardValidationResult:

    validator = WalkForwardValidator(
        minimum_windows=minimum_windows,
        require_positive_return=(
            require_positive_return
        ),
        require_positive_window_rate=(
            require_positive_window_rate
        ),
    )

    validation_result = validator.validate(
        result
    )

    validator.print_report(
        validation_result
    )

    return validation_result


__all__ = [
    "WalkForwardValidationResult",
    "WalkForwardValidator",
    "validate_walk_forward",
]