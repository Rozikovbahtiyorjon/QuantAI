"""
=========================================================
QuantAI Professional v5
Paper Trading Runner

Connects Strategy Engine with PaperTradingEngine.

Pipeline:

    Market Data
          ↓
      Strategy
          ↓
     SignalResult
          ↓
  PaperTradingRunner
          ↓
  PaperTradingEngine

This module does NOT:
    - connect to Binance
    - execute real orders
    - train ML models
    - calculate indicators
    - modify Strategy logic

It only connects strategy signals
to virtual paper execution.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.paper_trading_engine import (
    PaperTrade,
    PaperTradingEngine,
)
from src.strategy import (
    SignalResult,
    generate_signal_result,
)


# =========================================================
# RESULT
# =========================================================

@dataclass
class PaperTradingStepResult:
    """
    Result of processing one market-data step.
    """

    signal: SignalResult

    trade: Optional[PaperTrade]

    position_opened: bool

    position_closed: bool


# =========================================================
# RUNNER
# =========================================================

class PaperTradingRunner:
    """
    Connect Strategy Engine with PaperTradingEngine.

    Strategy decides WHAT to do.

    PaperTradingEngine decides HOW the
    virtual position is executed.
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
        commission: float = 0.0004,
        quantity: float = 1.0,
    ) -> None:

        if quantity <= 0:
            raise ValueError(
                "quantity must be greater than zero."
            )

        self.engine = PaperTradingEngine(
            initial_balance=initial_balance,
            commission=commission,
        )

        self.quantity = float(quantity)

    # =====================================================
    # PROCESS SIGNAL
    # =====================================================

    def process_signal(
        self,
        signal: SignalResult,
    ) -> PaperTradingStepResult:
        """
        Process one Strategy signal.

        BUY  -> open LONG
        SELL -> open SHORT
        HOLD -> do nothing
        """

        if not isinstance(
            signal,
            SignalResult,
        ):
            raise TypeError(
                "signal must be SignalResult."
            )

        position_opened = False
        position_closed = False

        trade = None

        # -------------------------------------------------
        # HOLD
        # -------------------------------------------------

        if signal.signal == "HOLD":

            return PaperTradingStepResult(
                signal=signal,
                trade=None,
                position_opened=False,
                position_closed=False,
            )

        # -------------------------------------------------
        # BUY
        # -------------------------------------------------

        if signal.signal == "BUY":

            if not self.engine.has_position:

                self.engine.open_position(
                    side="LONG",
                    price=signal.entry,
                    quantity=self.quantity,
                )

                position_opened = True

                return PaperTradingStepResult(
                    signal=signal,
                    trade=None,
                    position_opened=True,
                    position_closed=False,
                )

            return PaperTradingStepResult(
                signal=signal,
                trade=None,
                position_opened=False,
                position_closed=False,
            )

        # -------------------------------------------------
        # SELL
        # -------------------------------------------------

        if signal.signal == "SELL":

            if not self.engine.has_position:

                self.engine.open_position(
                    side="SHORT",
                    price=signal.entry,
                    quantity=self.quantity,
                )

                position_opened = True

                return PaperTradingStepResult(
                    signal=signal,
                    trade=None,
                    position_opened=True,
                    position_closed=False,
                )

            return PaperTradingStepResult(
                signal=signal,
                trade=None,
                position_opened=False,
                position_closed=False,
            )

        # -------------------------------------------------
        # UNKNOWN SIGNAL
        # -------------------------------------------------

        raise ValueError(
            f"Unsupported signal: {signal.signal}"
        )

    # =====================================================
    # CLOSE
    # =====================================================

    def close_position(
        self,
        price: float,
        signal: SignalResult | None = None,
    ) -> PaperTradingStepResult:
        """
        Close the current paper position.

        If signal is not supplied, a HOLD signal
        is created for the result object.
        """

        if price <= 0:
            raise ValueError(
                "price must be greater than zero."
            )

        if signal is None:

            signal = SignalResult(
                signal="HOLD",
                entry=price,
            )

        trade = self.engine.close_position(
            price=price,
        )

        return PaperTradingStepResult(
            signal=signal,
            trade=trade,
            position_opened=False,
            position_closed=True,
        )

    # =====================================================
    # PROCESS MARKET DATA
    # =====================================================

    def process_dataframe(
        self,
        df: pd.DataFrame,
    ) -> list[PaperTradingStepResult]:
        """
        Generate Strategy signals from a DataFrame
        and process them sequentially.

        This method intentionally keeps Strategy
        responsible for signal generation.
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

        results: list[
            PaperTradingStepResult
        ] = []

        for end in range(
            1,
            len(df) + 1,
        ):

            window = df.iloc[
                :end
            ].copy()

            signal = generate_signal_result(
                window
            )

            result = self.process_signal(
                signal
            )

            results.append(
                result
            )

        return results

    # =====================================================
    # ACCOUNT STATE
    # =====================================================

    @property
    def balance(self) -> float:
        """
        Current paper account balance.
        """

        return self.engine.balance

    @property
    def has_position(self) -> bool:
        """
        Whether a paper position is open.
        """

        return self.engine.has_position

    @property
    def realized_profit(self) -> float:
        """
        Total realized paper profit.
        """

        return self.engine.realized_profit

    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:
        """
        Reset paper trading state.
        """

        self.engine.reset()


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "PaperTradingStepResult",
    "PaperTradingRunner",
]