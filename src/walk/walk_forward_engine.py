"""
QuantAI Walk-Forward Engine v2.

Sequential out-of-sample validation engine.

IMPORTANT COMPATIBILITY CONTRACT

generate_windows() returns exactly:

(
    window_number,
    train_df,
    test_df,
)

The engine itself does NOT:

- connect to Binance;
- execute real orders;
- calculate indicators;
- modify Strategy;
- directly train ML models unless an optional
  train_callback is explicitly provided.

Default behavior remains compatible with the
previous WalkForwardEngine implementation.

Architecture:

Historical Data
       |
       v
WalkForwardEngine
       |
       +--> TRAIN WINDOW
       |       |
       |       +--> optional train_callback
       |
       +--> TEST WINDOW
               |
               v
        BacktestEngine
               |
               v
        BacktestResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd

from src.backtest_engine import (
    BacktestEngine,
    BacktestResult,
)


# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_TRAIN_SIZE = 500
DEFAULT_TEST_SIZE = 100
MINIMUM_WINDOW_SIZE = 1
DEFAULT_INITIAL_BALANCE = 1000.0


# =========================================================
# TYPES
# =========================================================

WindowTuple = Tuple[
    int,
    pd.DataFrame,
    pd.DataFrame,
]

TrainCallback = Callable[
    [pd.DataFrame, pd.DataFrame, int],
    Any,
]


# =========================================================
# WINDOW RESULT
# =========================================================

@dataclass
class WalkForwardWindowResult:
    """
    Result of one completed Walk-Forward window.
    """

    window_id: int

    train_start: int
    train_end: int

    test_start: int
    test_end: int

    train_size: int
    test_size: int

    backtest_result: BacktestResult

    # Optional metadata for future ML integration.
    model_result: Any = None

    @property
    def window_number(self) -> int:
        """
        Backward-compatible alias.
        """
        return self.window_id


# =========================================================
# COMPLETE RESULT
# =========================================================

@dataclass
class WalkForwardResult:
    """
    Complete Walk-Forward analysis result.
    """

    initial_balance: float
    final_balance: float
    net_profit: float

    total_trades: int
    winning_trades: int
    losing_trades: int

    win_rate: float

    windows: List[
        WalkForwardWindowResult
    ] = field(
        default_factory=list
    )

    @property
    def window_results(
        self,
    ) -> List[WalkForwardWindowResult]:
        """
        Backward-compatible alias.
        """
        return self.windows

    @property
    def total_windows(self) -> int:
        """
        Number of completed Walk-Forward windows.
        """
        return len(self.windows)


# =========================================================
# ENGINE
# =========================================================

class WalkForwardEngine:
    """
    Sequential out-of-sample validation engine.

    Example:

        train_size = 10
        test_size = 5
        step_size = 5

        Window 1:
            TRAIN 0:10
            TEST 10:15

        Window 2:
            TRAIN 5:15
            TEST 15:20

        Window 3:
            TRAIN 10:20
            TEST 20:25

    The train data is available for optional model
    training through train_callback.

    Without train_callback, behavior remains compatible
    with the previous implementation.
    """

    def __init__(
        self,
        train_size: int = DEFAULT_TRAIN_SIZE,
        test_size: int = DEFAULT_TEST_SIZE,
        step_size: Optional[int] = None,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        train_callback: Optional[
            TrainCallback
        ] = None,
    ) -> None:

        # -------------------------------------------------
        # VALIDATE TRAIN SIZE
        # -------------------------------------------------

        if type(train_size) is not int:
            raise TypeError(
                "train_size must be an integer."
            )

        if train_size < MINIMUM_WINDOW_SIZE:
            raise ValueError(
                "train_size must be greater than zero."
            )

        # -------------------------------------------------
        # VALIDATE TEST SIZE
        # -------------------------------------------------

        if type(test_size) is not int:
            raise TypeError(
                "test_size must be an integer."
            )

        if test_size < MINIMUM_WINDOW_SIZE:
            raise ValueError(
                "test_size must be greater than zero."
            )

        # -------------------------------------------------
        # VALIDATE STEP SIZE
        # -------------------------------------------------

        if step_size is not None:

            if type(step_size) is not int:
                raise TypeError(
                    "step_size must be an integer."
                )

            if step_size <= 0:
                raise ValueError(
                    "step_size must be greater than zero."
                )

        # -------------------------------------------------
        # VALIDATE BALANCE
        # -------------------------------------------------

        if initial_balance <= 0:
            raise ValueError(
                "initial_balance must be greater than zero."
            )

        # -------------------------------------------------
        # VALIDATE CALLBACK
        # -------------------------------------------------

        if train_callback is not None and not callable(
            train_callback
        ):
            raise TypeError(
                "train_callback must be callable."
            )

        # -------------------------------------------------
        # STORE CONFIGURATION
        # -------------------------------------------------

        self.train_size = train_size
        self.test_size = test_size

        self.step_size = (
            step_size
            if step_size is not None
            else test_size
        )

        self.initial_balance = float(
            initial_balance
        )

        self.train_callback = train_callback

        # -------------------------------------------------
        # LAST RESULT
        # -------------------------------------------------

        self._result: Optional[
            WalkForwardResult
        ] = None

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate_data(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate Walk-Forward input data.

        Requirements:

            - pandas DataFrame
            - non-empty
            - enough rows for one complete
              train + test window
        """

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "WalkForwardEngine requires "
                "a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Walk-forward data is empty."
            )

        minimum_required = (
            self.train_size
            + self.test_size
        )

        if len(df) < minimum_required:
            raise ValueError(
                "Not enough rows for walk-forward "
                f"analysis. Required at least "
                f"{minimum_required}, received "
                f"{len(df)}."
            )

    # =====================================================
    # WINDOW GENERATOR
    # =====================================================

    def generate_windows(
        self,
        df: pd.DataFrame,
    ) -> List[WindowTuple]:
        """
        Generate sequential train/test windows.

        COMPATIBILITY CONTRACT:

        Every returned item is exactly:

            (
                window_number,
                train_df,
                test_df,
            )

        Original DataFrame indexes are preserved.

        Returned DataFrames are copies.

        Incomplete final test windows are excluded.

        step_size defines the starting position
        of the next training window.
        """

        self.validate_data(df)

        windows: List[WindowTuple] = []

        total_rows = len(df)

        window_number = 1
        start = 0

        while True:

            train_start = start

            train_end = (
                train_start
                + self.train_size
            )

            test_start = train_end

            test_end = (
                test_start
                + self.test_size
            )

            # ---------------------------------------------
            # STOP IF TRAIN WINDOW IS INCOMPLETE
            # ---------------------------------------------

            if train_end > total_rows:
                break

            # ---------------------------------------------
            # STOP IF TEST WINDOW IS INCOMPLETE
            # ---------------------------------------------

            if test_end > total_rows:
                break

            # ---------------------------------------------
            # CREATE COPIES
            # ---------------------------------------------

            train_df = (
                df.iloc[
                    train_start:train_end
                ]
                .copy()
            )

            test_df = (
                df.iloc[
                    test_start:test_end
                ]
                .copy()
            )

            # ---------------------------------------------
            # SAFETY CHECKS
            # ---------------------------------------------

            if len(train_df) != self.train_size:
                break

            if len(test_df) != self.test_size:
                break

            # ---------------------------------------------
            # APPEND
            # ---------------------------------------------

            windows.append(
                (
                    window_number,
                    train_df,
                    test_df,
                )
            )

            # ---------------------------------------------
            # ADVANCE
            # ---------------------------------------------

            start += self.step_size

            window_number += 1

        return windows

    # =====================================================
    # WINDOW BOUNDARIES
    # =====================================================

    def _get_window_boundaries(
        self,
        window_number: int,
    ) -> Tuple[int, int, int, int]:
        """
        Return:

            train_start,
            train_end,
            test_start,
            test_end

        for a given 1-based window number.
        """

        if type(window_number) is not int:
            raise TypeError(
                "window_number must be an integer."
            )

        if window_number <= 0:
            raise ValueError(
                "window_number must be greater than zero."
            )

        start = (
            (window_number - 1)
            * self.step_size
        )

        train_start = start

        train_end = (
            train_start
            + self.train_size
        )

        test_start = train_end

        test_end = (
            test_start
            + self.test_size
        )

        return (
            train_start,
            train_end,
            test_start,
            test_end,
        )

    # =====================================================
    # SINGLE WINDOW
    # =====================================================

    def run_window(
        self,
        df: pd.DataFrame,
        window_id: int,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int,
        initial_balance: float,
    ) -> WalkForwardWindowResult:
        """
        Run one Walk-Forward test window.

        train_df is available for optional ML training.

        BacktestEngine receives only test_df.

        This preserves the important separation:

            TRAIN
              |
              v
        optional model update
              |
              v
            TEST
              |
              v
        BacktestEngine
        """

        # -------------------------------------------------
        # VALIDATE DATA
        # -------------------------------------------------

        self.validate_data(df)

        # -------------------------------------------------
        # VALIDATE WINDOW ID
        # -------------------------------------------------

        if type(window_id) is not int:
            raise TypeError(
                "window_id must be an integer."
            )

        if window_id <= 0:
            raise ValueError(
                "window_id must be greater than zero."
            )

        # -------------------------------------------------
        # VALIDATE BALANCE
        # -------------------------------------------------

        if initial_balance <= 0:
            raise ValueError(
                "initial_balance must be greater than zero."
            )

        # -------------------------------------------------
        # VALIDATE BOUNDARIES
        # -------------------------------------------------

        if train_start < 0:
            raise ValueError(
                "train_start cannot be negative."
            )

        if train_end <= train_start:
            raise ValueError(
                "Train window cannot be empty."
            )

        if test_start < train_end:
            raise ValueError(
                "Test window must start after "
                "the training window."
            )

        if test_end <= test_start:
            raise ValueError(
                "Test window cannot be empty."
            )

        if test_end > len(df):
            raise ValueError(
                "Test window exceeds available data."
            )

        # -------------------------------------------------
        # SIZES
        # -------------------------------------------------

        train_size = (
            train_end
            - train_start
        )

        test_size = (
            test_end
            - test_start
        )

        # -------------------------------------------------
        # EXTRACT TRAIN
        # -------------------------------------------------

        train_df = (
            df.iloc[
                train_start:train_end
            ]
            .copy()
        )

        # -------------------------------------------------
        # EXTRACT TEST
        # -------------------------------------------------

        test_df = (
            df.iloc[
                test_start:test_end
            ]
            .copy()
        )

        # -------------------------------------------------
        # SAFETY CHECKS
        # -------------------------------------------------

        if len(train_df) != train_size:
            raise ValueError(
                "Generated training window has "
                "an unexpected size."
            )

        if len(test_df) != test_size:
            raise ValueError(
                "Generated testing window has "
                "an unexpected size."
            )

        # -------------------------------------------------
        # OPTIONAL MODEL TRAINING
        # -------------------------------------------------

        model_result = None

        if self.train_callback is not None:

            model_result = self.train_callback(
                train_df,
                test_df,
                window_id,
            )

        # -------------------------------------------------
        # BACKTEST
        # -------------------------------------------------

        backtest = BacktestEngine(
            initial_balance=initial_balance,
        )

        backtest.minimum_rows = test_size

        # Auto-add indicators if test data is raw OHLCV
        if "atr" not in test_df.columns:
            from src.indicators import add_indicators

            test_df = add_indicators(test_df)

        backtest_result = backtest.run(
            test_df
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        return WalkForwardWindowResult(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_size=train_size,
            test_size=test_size,
            backtest_result=backtest_result,
            model_result=model_result,
        )

    # =====================================================
    # RUN
    # =====================================================

    def run(
        self,
        df: pd.DataFrame,
    ) -> WalkForwardResult:
        """
        Run complete Walk-Forward analysis.

        The final balance from one test window becomes
        the initial balance of the next test window.

        The default behavior remains compatible with
        the previous implementation.
        """

        self.validate_data(df)

        generated_windows = (
            self.generate_windows(df)
        )

        if not generated_windows:
            raise ValueError(
                "No valid walk-forward windows "
                "could be generated."
            )

        results: List[
            WalkForwardWindowResult
        ] = []

        current_balance = (
            self.initial_balance
        )

        # -------------------------------------------------
        # RUN WINDOWS
        # -------------------------------------------------

        for (
            window_number,
            train_df,
            test_df,
        ) in generated_windows:

            (
                train_start,
                train_end,
                test_start,
                test_end,
            ) = self._get_window_boundaries(
                window_number
            )

            # ---------------------------------------------
            # SAFETY: MATCH GENERATED WINDOWS
            # ---------------------------------------------

            if len(train_df) != self.train_size:
                raise ValueError(
                    "Generated train window size "
                    "does not match engine configuration."
                )

            if len(test_df) != self.test_size:
                raise ValueError(
                    "Generated test window size "
                    "does not match engine configuration."
                )

            # ---------------------------------------------
            # RUN WINDOW
            # ---------------------------------------------

            window_result = self.run_window(
                df=df,
                window_id=window_number,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                initial_balance=current_balance,
            )

            results.append(
                window_result
            )

            # ---------------------------------------------
            # ROLL BALANCE FORWARD
            # ---------------------------------------------

            current_balance = float(
                window_result
                .backtest_result
                .final_balance
            )

        # -------------------------------------------------
        # FINAL BALANCE
        # -------------------------------------------------

        final_balance = current_balance

        # -------------------------------------------------
        # NET PROFIT
        # -------------------------------------------------

        net_profit = (
            final_balance
            - self.initial_balance
        )

        # -------------------------------------------------
        # AGGREGATE TRADES
        # -------------------------------------------------

        total_trades = sum(
            window.backtest_result.total_trades
            for window in results
        )

        winning_trades = sum(
            window.backtest_result.winning_trades
            for window in results
        )

        losing_trades = sum(
            window.backtest_result.losing_trades
            for window in results
        )

        # -------------------------------------------------
        # AGGREGATE WIN RATE
        # -------------------------------------------------

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

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        self._result = WalkForwardResult(
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            net_profit=round(
                net_profit,
                8,
            ),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            windows=results,
        )

        return self._result

    # =====================================================
    # RESULT PROPERTY
    # =====================================================

    @property
    def result(
        self,
    ) -> Optional[WalkForwardResult]:
        """
        Return the latest completed result.

        Returns None before run() is executed.
        """

        return self._result

    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:
        """
        Clear the latest Walk-Forward result.

        Configuration remains unchanged.
        """

        self._result = None

    # =====================================================
    # REPORT
    # =====================================================

    @staticmethod
    def print_report(
        result: WalkForwardResult,
    ) -> None:
        """
        Print a concise Walk-Forward report.
        """

        if not isinstance(
            result,
            WalkForwardResult,
        ):
            raise TypeError(
                "result must be WalkForwardResult."
            )

        print()

        print("=" * 70)
        print("QUANTAI WALK-FORWARD REPORT")
        print("=" * 70)

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

        print("-" * 70)

        print(
            f"Windows         : "
            f"{result.total_windows}"
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

        print("-" * 70)

        for window in result.windows:

            backtest = (
                window.backtest_result
            )

            model_status = (
                "trained"
                if window.model_result is not None
                else "not_trained"
            )

            print(
                f"Window "
                f"{window.window_number}: "
                f"TRAIN="
                f"{window.train_start}:"
                f"{window.train_end} | "
                f"TEST="
                f"{window.test_start}:"
                f"{window.test_end} | "
                f"train_size="
                f"{window.train_size} | "
                f"test_size="
                f"{window.test_size} | "
                f"trades="
                f"{backtest.total_trades} | "
                f"profit="
                f"{backtest.net_profit:.2f} | "
                f"win_rate="
                f"{backtest.win_rate:.2f}% | "
                f"model="
                f"{model_status}"
            )

        print("=" * 70)


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def run_walk_forward(
    df: pd.DataFrame,
    train_size: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    step_size: Optional[int] = None,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    train_callback: Optional[
        TrainCallback
    ] = None,
) -> WalkForwardResult:
    """
    Convenience wrapper.

    By default behaves exactly like the previous
    run_walk_forward() interface.

    train_callback is optional and reserved for
    future ML Walk-Forward integration.
    """

    engine = WalkForwardEngine(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        initial_balance=initial_balance,
        train_callback=train_callback,
    )

    result = engine.run(df)

    engine.print_report(result)

    return result


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "DEFAULT_TRAIN_SIZE",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_INITIAL_BALANCE",
    "MINIMUM_WINDOW_SIZE",
    "WindowTuple",
    "TrainCallback",
    "WalkForwardWindowResult",
    "WalkForwardResult",
    "WalkForwardEngine",
    "run_walk_forward",
]