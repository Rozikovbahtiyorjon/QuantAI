"""
=========================================================
QuantAI Professional v5
Paper Trading Pipeline

High-level orchestration layer for paper trading.

Pipeline:

    OHLCV DataFrame
          ↓
    PaperTradingPipeline
          ↓
    PaperTradingSession
          ↓
    PaperTradingRunner
          ↓
    Strategy + PaperTradingEngine
          ↓
    PaperTradingPipelineResult

This module does NOT:
    - connect to Binance
    - execute real orders
    - calculate indicators
    - train ML models
    - modify Strategy logic
    - modify PaperTradingEngine
    - modify PaperTradingRunner
    - modify PaperTradingSession

It only orchestrates a complete paper-trading run.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.paper_trading_session import (
    PaperTradingSession,
    PaperTradingSessionResult,
)


# =========================================================
# PIPELINE RESULT
# =========================================================

@dataclass
class PaperTradingPipelineResult:
    """
    Result of one complete paper-trading pipeline run.
    """

    session_result: PaperTradingSessionResult

    initial_balance: float

    final_balance: float

    realized_profit: float

    total_steps: int

    opened_positions: int

    closed_positions: int

    return_percent: float


# =========================================================
# PIPELINE
# =========================================================

class PaperTradingPipeline:
    """
    High-level paper-trading orchestration.

    The pipeline owns a PaperTradingSession and delegates
    actual processing to the existing session/runner stack.
    """

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

        self.session = PaperTradingSession(
            initial_balance=initial_balance,
            commission=commission,
            quantity=quantity,
        )

        self._result: Optional[
            PaperTradingPipelineResult
        ] = None

    # =====================================================
    # RUN
    # =====================================================

    def run(
        self,
        df: pd.DataFrame,
    ) -> PaperTradingPipelineResult:
        """
        Execute one complete paper-trading pipeline.

        Parameters
        ----------
        df:
            Market OHLCV DataFrame.

        Returns
        -------
        PaperTradingPipelineResult
            Aggregated pipeline result.
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

        session_result = self.session.run(
            df
        )

        self._result = (
            self._build_result(
                session_result
            )
        )

        return self._result

    # =====================================================
    # BUILD RESULT
    # =====================================================

    @staticmethod
    def _build_result(
        session_result: PaperTradingSessionResult,
    ) -> PaperTradingPipelineResult:
        """
        Convert a session result into a pipeline result.
        """

        initial_balance = float(
            session_result.initial_balance
        )

        final_balance = float(
            session_result.final_balance
        )

        if initial_balance > 0:

            return_percent = round(
                (
                    (
                        final_balance
                        - initial_balance
                    )
                    / initial_balance
                )
                * 100.0,
                2,
            )

        else:

            return_percent = 0.0

        return PaperTradingPipelineResult(
            session_result=session_result,

            initial_balance=initial_balance,

            final_balance=final_balance,

            realized_profit=float(
                session_result.realized_profit
            ),

            total_steps=int(
                session_result.total_steps
            ),

            opened_positions=int(
                session_result.opened_positions
            ),

            closed_positions=int(
                session_result.closed_positions
            ),

            return_percent=return_percent,
        )

    # =====================================================
    # RESULT
    # =====================================================

    @property
    def result(
        self,
    ) -> PaperTradingPipelineResult | None:
        """
        Return the latest pipeline result.

        Returns None before the first run.
        """

        return self._result

    # =====================================================
    # ACCOUNT STATE
    # =====================================================

    @property
    def balance(self) -> float:
        """
        Current paper balance.
        """

        return self.session.balance

    @property
    def has_position(self) -> bool:
        """
        Whether a paper position is currently open.
        """

        return self.session.has_position

    @property
    def realized_profit(self) -> float:
        """
        Current realized paper profit.
        """

        return self.session.realized_profit

    @property
    def steps(self):
        """
        Return processed paper-trading steps.
        """

        return self.session.steps

    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:
        """
        Reset the complete paper-trading pipeline.
        """

        self.session.reset()

        self._result = None


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "PaperTradingPipelineResult",
    "PaperTradingPipeline",
]
