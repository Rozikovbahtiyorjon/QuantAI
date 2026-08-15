from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class RegimeWindow:
    regime: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass(frozen=True)
class RegimeWalkForwardResult:
    total_windows: int
    regimes: tuple[str, ...]
    windows_by_regime: dict[str, int]
    rows_by_regime: dict[str, int]


class RegimeWalkForward:
    """
    Splits market data into regime-specific walk-forward windows.

    The engine does not modify the existing WalkForwardEngine.
    It only prepares regime-aware train/test windows for downstream
    validation and backtesting.
    """

    VALID_REGIMES = (
        "BULL",
        "BEAR",
        "RANGE",
        "HIGH_VOLATILITY",
        "LOW_VOLATILITY",
        "NEUTRAL",
    )

    def __init__(
        self,
        regime_column: str = "regime",
    ) -> None:
        if not isinstance(regime_column, str):
            raise TypeError(
                "regime_column must be a string."
            )

        if not regime_column.strip():
            raise ValueError(
                "regime_column cannot be empty."
            )

        self.regime_column = regime_column.strip()
        self._result: RegimeWalkForwardResult | None = None

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "DataFrame cannot be empty."
            )

        if self.regime_column not in df.columns:
            raise ValueError(
                "DataFrame is missing required regime column: "
                + self.regime_column
            )

    @staticmethod
    def _normalize_regime(
        value: object,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                "regime values must be strings."
            )

        regime = value.strip().upper()

        if not regime:
            raise ValueError(
                "regime values cannot be empty."
            )

        return regime

    def generate_windows(
        self,
        df: pd.DataFrame,
        train_size: int,
        test_size: int,
    ) -> list[RegimeWindow]:
        self._validate_dataframe(df)

        if train_size <= 0:
            raise ValueError(
                "train_size must be greater than zero."
            )

        if test_size <= 0:
            raise ValueError(
                "test_size must be greater than zero."
            )

        if train_size + test_size > len(df):
            raise ValueError(
                "train_size + test_size cannot exceed "
                "the number of rows."
            )

        data = df.copy()

        data[self.regime_column] = data[
            self.regime_column
        ].map(self._normalize_regime)

        windows: list[RegimeWindow] = []

        start = 0

        while start + train_size + test_size <= len(data):
            train_end = start + train_size
            test_end = train_end + test_size

            train_df = data.iloc[
                start:train_end
            ].copy()

            test_df = data.iloc[
                train_end:test_end
            ].copy()

            regimes = set(
                train_df[self.regime_column]
            ).union(
                set(test_df[self.regime_column])
            )

            for regime in sorted(regimes):
                regime_train = train_df[
                    train_df[self.regime_column]
                    == regime
                ].copy()

                regime_test = test_df[
                    test_df[self.regime_column]
                    == regime
                ].copy()

                if regime_train.empty and regime_test.empty:
                    continue

                windows.append(
                    RegimeWindow(
                        regime=regime,
                        train_df=regime_train,
                        test_df=regime_test,
                    )
                )

            start += test_size

        return windows

    def run(
        self,
        df: pd.DataFrame,
        train_size: int,
        test_size: int,
    ) -> RegimeWalkForwardResult:
        windows = self.generate_windows(
            df=df,
            train_size=train_size,
            test_size=test_size,
        )

        windows_by_regime: dict[str, int] = {}
        rows_by_regime: dict[str, int] = {}

        for window in windows:
            windows_by_regime[window.regime] = (
                windows_by_regime.get(
                    window.regime,
                    0,
                )
                + 1
            )

            rows_by_regime[window.regime] = (
                rows_by_regime.get(
                    window.regime,
                    0,
                )
                + len(window.train_df)
                + len(window.test_df)
            )

        regimes = tuple(
            sorted(windows_by_regime)
        )

        result = RegimeWalkForwardResult(
            total_windows=len(windows),
            regimes=regimes,
            windows_by_regime=windows_by_regime,
            rows_by_regime=rows_by_regime,
        )

        self._result = result

        return result

    @property
    def result(
        self,
    ) -> RegimeWalkForwardResult | None:
        return self._result

    def reset(self) -> None:
        self._result = None


__all__ = [
    "RegimeWindow",
    "RegimeWalkForwardResult",
    "RegimeWalkForward",
]