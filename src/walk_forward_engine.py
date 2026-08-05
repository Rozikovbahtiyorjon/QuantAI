"""
=========================================================
QuantAI Professional v5
Walk-Forward Engine

Walk-forward validation orchestration.

Pipeline:

    Historical Data
          ↓
    WalkForwardEngine
          ↓
    Train Window
          ↓
    Test Window
          ↓
    BacktestEngine
          ↓
    Out-of-Sample Result
          ↓
    Next Window
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from src.backtest_engine import (
    BacktestEngine,
    BacktestResult,
)


# =========================================================
# RESULT
# =========================================================

@dataclass
class WalkForwardResult:
    """
    Result of one walk-forward test window.
    """

    window_number: int

    train_start: object
    train_end: object

    test_start: object
    test_end: object

    backtest_result: BacktestResult


# =========================================================
# ENGINE
# =========================================================

class WalkForwardEngine:
    """
    Walk-forward validation controller.

    The engine does NOT:

    - load market data
    - calculate indicators
    - modify Strategy
    - modify TradeEngine
    - train ML models

    It only creates sequential train/test windows
    and runs BacktestEngine on the out-of-sample
    test window.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        initial_balance: float | None = None,
    ) -> None:

        if train_size <= 0:
            raise ValueError(
                "train_size must be greater than zero."
            )

        if test_size <= 0:
            raise ValueError(
                "test_size must be greater than zero."
            )

        if step_size is None:
           step_size = test_size

        if step_size <= 0:
            raise ValueError(
                "step_size must be greater than zero."
            )

        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.initial_balance = initial_balance

    # =====================================================
    # VALIDATION
    # =====================================================

    @staticmethod
    def validate_data(
        df: pd.DataFrame,
    ) -> None:
        """
        Validate historical DataFrame.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "WalkForwardEngine requires "
                "a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Walk-forward data is empty."
            )

    # =====================================================
    # WINDOW GENERATOR
    # =====================================================

    def generate_windows(
        self,
        df: pd.DataFrame,
    ):
        """
        Generate sequential train/test windows.

        Yields:

            train_df,
            test_df
        """

        self.validate_data(df)

        data = df.reset_index(drop=True)

        start = 0
        window_number = 1

        while (
            start
            + self.train_size
            + self.test_size
            <= len(data)
        ):

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
        
            train_df = data.iloc[
                train_start:train_end
            ].copy()
        
            test_df = data.iloc[
                test_start:test_end
            ].copy()
        
            yield (
                window_number,
                train_df,
                test_df,
            )
        
            window_number += 1
        
            # IMPORTANT:
            # Rolling walk-forward window.
            # The next TRAIN starts step_size rows after
            # the previous TRAIN start.
            start += self.step_size
        
    # =====================================================
    # RUN
    # =====================================================

    def run(
        self,
        df: pd.DataFrame,
    ) -> List[WalkForwardResult]:
        """
        Run walk-forward validation.
        """

        self.validate_data(df)

        results: List[WalkForwardResult] = []

        for (
            window_number,
            train_df,
            test_df,
        ) in self.generate_windows(df):

            backtest_engine = BacktestEngine(
                initial_balance=self.initial_balance,
            )

            backtest_result = backtest_engine.run(
                test_df
            )

            result = WalkForwardResult(
                window_number=window_number,

                train_start=train_df.iloc[0]["timestamp"],
                train_end=train_df.iloc[-1]["timestamp"],

                test_start=test_df.iloc[0]["timestamp"],
                test_end=test_df.iloc[-1]["timestamp"],

                backtest_result=backtest_result,
            )

            results.append(result)

        return results


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "WalkForwardResult",
    "WalkForwardEngine",
]