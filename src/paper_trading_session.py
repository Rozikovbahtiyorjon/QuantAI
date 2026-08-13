"""
QuantAI Professional v5
Paper Trading Session

Orchestrates sequential paper trading over market data.

Pipeline:

OHLCV DataFrame
↓
PaperMarketData
↓
Sequential market rows
↓
Strategy
↓
SignalResult
↓
PaperTradingRunner
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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from src.paper_market_data import PaperMarketData
from src.paper_trading_runner import (
    PaperTradingRunner,
    PaperTradingStepResult,
)
from src import paper_trading_runner as _paper_trading_runner


def generate_signal_result(df: pd.DataFrame):
    """
    Compatibility proxy for Strategy signal generation.

    The proxy is intentionally kept at module level so existing tests can
    monkeypatch ``src.paper_trading_session.generate_signal_result``.

    The actual call is delegated dynamically to ``paper_trading_runner`` so
    integrations that monkeypatch
    ``src.paper_trading_runner.generate_signal_result`` continue to work.
    """

    return _paper_trading_runner.generate_signal_result(df)


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


class PaperTradingSession:
    """
    Sequential paper-trading session.

    Market data is consumed through PaperMarketData.
    Strategy signals are generated from the accumulated
    market-data window and passed to PaperTradingRunner.
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

        self._steps: List[PaperTradingStepResult] = []

        self._market_data: PaperMarketData | None = None

    def run(
        self,
        df: pd.DataFrame,
    ) -> PaperTradingSessionResult:
        """
        Run a complete paper-trading session.

        Market data is consumed sequentially through PaperMarketData.

        For every market-data row, the accumulated DataFrame window is
        passed to Strategy. The resulting SignalResult is processed by
        PaperTradingRunner.
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

        market_data = PaperMarketData(df)

        self._market_data = market_data
        self._steps = []

        accumulated_rows: list[pd.Series] = []

        for row in market_data:
            accumulated_rows.append(row)

            window = pd.DataFrame(
                accumulated_rows
            ).reset_index(drop=True)

            signal = generate_signal_result(
                window
            )

            step = self.runner.process_signal(
                signal
            )

            self._steps.append(step)

        return self.result

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

    @property
    def balance(
        self,
    ) -> float:
        """
        Current paper balance.
        """

        return self.runner.balance

    @property
    def has_position(
        self,
    ) -> bool:
        """
        Whether a paper position is open.
        """

        return self.runner.has_position

    @property
    def realized_profit(
        self,
    ) -> float:
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

    @property
    def market_data(
        self,
    ) -> PaperMarketData | None:
        """
        Return the current market-data provider.
        """

        return self._market_data

    def reset(
        self,
    ) -> None:
        """
        Reset the paper-trading session.
        """

        self.runner.reset()

        self._steps.clear()

        if self._market_data is not None:
            self._market_data.reset()


__all__ = [
    "PaperTradingSessionResult",
    "PaperTradingSession",
]