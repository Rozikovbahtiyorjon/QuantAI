"""
=========================================================
QuantAI Professional v5
Paper Trading Session

Orchestrates sequential paper trading over market data.

Pipeline:

    OHLCV DataFrame
          ↓
       Strategy
          ↓
    SignalResult
          ↓
  PaperTradingRunner
          ↓
  PaperTradingEngine
          ↓
    Trade History

This module does NOT:
    - connect to Binance
    - execute real orders
    - train ML models
    - calculate indicators
    - modify Strategy logic
    - modify PaperTradingEngine
    - modify PaperTradingRunner

It only controls the paper-trading session.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from src.paper_trading_runner import (
    PaperTradingRunner,
    PaperTradingStepResult,
)


# =========================================================
# SESSION RESULT
# =========================================================

@dataclass
class PaperTradingSessionResult:
    """
    Result of one complete paper-trading session.
    """

    steps: List[PaperTradingStepResult]

    initial_balance: float

    final_balance: float

    realized_profit: float

    total_steps: int

    opened_positions: int

    closed_positions: int


# =========================================================
# SESSION
# =========================================================

class PaperTradingSession:
    """
    Sequential paper-trading session.

    Receives market data and sends each step
    through PaperTradingRunner.
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
        commission: float = 0.0004,
        quantity: float = 1.0,
    ) -> None:

        self.runner = PaperTradingRunner(
            initial_balance=initial_balance,
            commission=commission,
            quantity=quantity,
        )

        self._steps: List[
            PaperTradingStepResult
        ] = []

    # =====================================================
    # RUN
    # =====================================================

    def run(
        self,
        df: pd.DataFrame,
    ) -> PaperTradingSessionResult:
        """
        Run a complete paper-trading session.

        The DataFrame is processed sequentially.
        """

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "DataFrame cannot be empty."
            )

        self._steps = (
            self.runner.process_dataframe(
                df
            )
        )

        return self.result

    # =====================================================
    # RESULT
    # =====================================================

    @property
    def result(
        self,
    ) -> PaperTradingSessionResult:
        """
        Return the current session result.
        """

        opened_positions = sum(
            1
            for step in self._steps
            if step.position_opened
        )

        closed_positions = sum(
            1
            for step in self._steps
            if step.position_closed
        )

        return PaperTradingSessionResult(
            steps=list(self._steps),

            initial_balance=(
                self.runner.engine.initial_balance
            ),

            final_balance=(
                self.runner.balance
            ),

            realized_profit=(
                self.runner.realized_profit
            ),

            total_steps=len(
                self._steps
            ),

            opened_positions=(
                opened_positions
            ),

            closed_positions=(
                closed_positions
            ),
        )

    # =====================================================
    # STATE
    # =====================================================

    @property
    def balance(self) -> float:
        """
        Current paper balance.
        """

        return self.runner.balance

    @property
    def has_position(self) -> bool:
        """
        Whether a paper position is open.
        """

        return self.runner.has_position

    @property
    def realized_profit(self) -> float:
        """
        Current realized profit.
        """

        return self.runner.realized_profit

    @property
    def steps(
        self,
    ) -> List[PaperTradingStepResult]:
        """
        Return processed steps.
        """

        return list(self._steps)

    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:
        """
        Reset the session.
        """

        self.runner.reset()

        self._steps.clear()


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "PaperTradingSessionResult",
    "PaperTradingSession",
]