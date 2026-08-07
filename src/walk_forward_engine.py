"""
QuantAI - Walk Forward Engine
=============================

Sequential out-of-sample backtesting engine.

IMPORTANT COMPATIBILITY CONTRACT
--------------------------------

generate_windows() returns exactly:

    (
        window_number,
        train_df,
        test_df,
    )

The engine does NOT:

    - connect to Binance
    - execute real orders
    - calculate indicators
    - modify Strategy
    - train ML models directly

It orchestrates sequential out-of-sample
backtesting using BacktestEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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

    windows: List[WalkForwardWindowResult] = field(
        default_factory=list
    )

    @property
    def window_results(
        self,
    ) -> List[WalkForwardWindowResult]:
        """
        Backward-compatible alias for windows.
        """
        return self.windows

    @property
    def total_windows(self) -> int:
        """
        Number of completed Walk-Forward windows.
        """
        return len(self.windows)


# =========================================================
# WINDOW TYPE
# =========================================================

WindowTuple = Tuple[
    int,
    pd.DataFrame,
    pd.DataFrame,
]


# =========================================================
# ENGINE
# =========================================================

class WalkForwardEngine:
    """
    Sequential out-of-sample backtesting engine.

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
    """

    def __init__(
        self,
        train_size: int = DEFAULT_TRAIN_SIZE,
        test_size: int = DEFAULT_TEST_SIZE,
        step_size: Optional[int] = None,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
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
    
            Each item is exactly:
    
                (
                    window_number,
                    train_df,
                    test_df,
                )
    
            Original DataFrame indexes are preserved.
    
            Returned DataFrames are copies.
    
            Incomplete final test windows are excluded.
    
            The step_size defines the starting position
            of the next training window.
    
            Example
            -------
    
            train_size = 5
            test_size = 5
            step_size = 10
    
            Window 1:
                TRAIN: 0:5
                TEST : 5:10
    
            Window 2:
                TRAIN: 10:15
                TEST : 15:20
    
            Window 3:
                TRAIN: 20:25
                TEST : 25:30
    
            Therefore, when step_size > test_size,
            unused rows between windows are preserved.
            """
    
            self.validate_data(df)
    
            windows: List[WindowTuple] = []
    
            window_number = 1
            total_rows = len(df)
    
            start = 0
    
            while True:
    
                # ---------------------------------------------
                # TRAIN WINDOW
                # ---------------------------------------------
    
                train_start = start
    
                train_end = (
                    train_start
                    + self.train_size
                )
    
                # ---------------------------------------------
                # TEST WINDOW
                # ---------------------------------------------
    
                test_start = train_end
    
                test_end = (
                    test_start
                    + self.test_size
                )
    
                # ---------------------------------------------
                # STOP ON INCOMPLETE TEST WINDOW
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
                # APPEND WINDOW
                # ---------------------------------------------
    
                windows.append(
                    (
                        window_number,
                        train_df,
                        test_df,
                    )
                )
    
                # ---------------------------------------------
                # ADVANCE TO NEXT WINDOW
                # ---------------------------------------------
    
                start += self.step_size
    
                window_number += 1
    
            return windows



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

        BacktestEngine receives only the test DataFrame.
        """

        # -------------------------------------------------
        # VALIDATE WINDOW
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
        # BACKTEST
        # -------------------------------------------------

        backtest = BacktestEngine(
            initial_balance=initial_balance
        )

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

            # ---------------------------------------------
            # ORIGINAL POSITIONS
            # ---------------------------------------------

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

            print(
                f"Window "
                f"{window.window_number}: "
                f"TRAIN="
                f"{window.train_start}:"
                f"{window.train_end} | "
                f"TEST="
                f"{window.test_start}:"
                f"{window.test_end} | "
                f"trades="
                f"{backtest.total_trades} | "
                f"profit="
                f"{backtest.net_profit:.2f} | "
                f"win_rate="
                f"{backtest.win_rate:.2f}%"
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
) -> WalkForwardResult:
    """
    Convenience wrapper.
    """

    engine = WalkForwardEngine(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        initial_balance=initial_balance,
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
    "WalkForwardWindowResult",
    "WalkForwardResult",
    "WalkForwardEngine",
    "run_walk_forward",
]
