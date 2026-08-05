"""
=========================================================
QuantAI Professional v5
Paper Market Data

Sequential market-data provider for paper trading.

This module does NOT:
    - generate signals
    - calculate indicators
    - execute trades
    - connect to Binance
    - train ML models

It only provides market rows sequentially.
=========================================================
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd


# =========================================================
# MARKET DATA PROVIDER
# =========================================================

class PaperMarketData:
    """
    Sequential market-data provider.

    Each iteration returns one market-data row as a
    pandas Series.
    """

    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:

        self.validate_data(data)

        self.data = data.reset_index(
            drop=True
        ).copy()

        self._position = 0

    # =====================================================
    # VALIDATION
    # =====================================================

    @staticmethod
    def validate_data(
        data: pd.DataFrame,
    ) -> None:
        """
        Validate market data.
        """

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "data cannot be empty."
            )

    # =====================================================
    # STATE
    # =====================================================

    @property
    def position(self) -> int:
        """
        Current zero-based position.
        """

        return self._position

    @property
    def total_rows(self) -> int:
        """
        Total number of market-data rows.
        """

        return len(self.data)

    @property
    def finished(self) -> bool:
        """
        Return True when all rows have been consumed.
        """

        return (
            self._position
            >= self.total_rows
        )

    # =====================================================
    # NEXT ROW
    # =====================================================

    def next(self) -> pd.Series:
        """
        Return the next market-data row.
        """

        if self.finished:
            raise StopIteration(
                "No more market data available."
            )

        row = self.data.iloc[
            self._position
        ]

        self._position += 1

        return row

    # =====================================================
    # ITERATOR
    # =====================================================

    def __iter__(
        self,
    ) -> Iterator[pd.Series]:

        while not self.finished:
            yield self.next()

    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:
        """
        Reset provider to the first row.
        """

        self._position = 0


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "PaperMarketData",
]