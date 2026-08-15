from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.paper_market_data import PaperMarketData
from src.paper_trading_session import (
    PaperTradingSession,
    PaperTradingSessionResult,
)


@dataclass(frozen=True)
class PaperTradingMarketSessionResult:
    session: PaperTradingSessionResult
    market_rows: int


class PaperTradingMarketSession:
    """Integrates sequential market data with paper trading."""

    def __init__(
        self,
        initial_balance: float = 1000.0,
        commission: float = 0.0004,
        quantity: float = 1.0,
    ) -> None:
        self.session = PaperTradingSession(
            initial_balance=initial_balance,
            commission=commission,
            quantity=quantity,
        )
        self._result: PaperTradingMarketSessionResult | None = None

    def run(
        self,
        data: pd.DataFrame,
    ) -> PaperTradingMarketSessionResult:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "data cannot be empty."
            )

        feed = PaperMarketData(data)
        rows: list[pd.Series] = []

        while not feed.finished:
            rows.append(feed.next())

        sequential_data = pd.DataFrame(rows).reset_index(
            drop=True
        )

        session_result = self.session.run(
            sequential_data
        )

        result = PaperTradingMarketSessionResult(
            session=session_result,
            market_rows=feed.total_rows,
        )

        self._result = result

        return result

    @property
    def result(
        self,
    ) -> PaperTradingMarketSessionResult | None:
        return self._result

    @property
    def balance(self) -> float:
        return self.session.balance

    @property
    def has_position(self) -> bool:
        return self.session.has_position

    @property
    def realized_profit(self) -> float:
        return self.session.realized_profit

    @property
    def steps(self):
        return self.session.steps

    def reset(self) -> None:
        self.session.reset()
        self._result = None


__all__ = [
    "PaperTradingMarketSessionResult",
    "PaperTradingMarketSession",
]