from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpenInterestDivergenceResult:
    signal: str
    strength: float
    price_change: float
    open_interest_change: float
    price_zscore: float
    open_interest_zscore: float
    divergence_score: float
    confidence: float


class OpenInterestDivergenceEngine:
    """
    Causal price/Open Interest divergence detector.

    The engine intentionally avoids centered rolling windows and future
    information so that the resulting features can be used in historical
    backtesting without look-ahead bias.

    Signals:
        BULLISH  -> price weakens while open interest strengthens
        BEARISH  -> price strengthens while open interest weakens
        NEUTRAL  -> no statistically meaningful divergence
    """

    REQUIRED_COLUMNS = ("close", "open_interest")

    def __init__(
        self,
        lookback: int = 5,
        zscore_window: int = 50,
        min_divergence: float = 0.10,
        min_confidence: float = 0.25,
    ) -> None:
        self.lookback = int(lookback)
        self.zscore_window = int(zscore_window)
        self.min_divergence = float(min_divergence)
        self.min_confidence = float(min_confidence)

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1.")

        if self.zscore_window < 2:
            raise ValueError("zscore_window must be >= 2.")

        if self.min_divergence < 0:
            raise ValueError("min_divergence must be >= 0.")

        if self.min_confidence < 0 or self.min_confidence > 1:
            raise ValueError("min_confidence must be between 0 and 1.")

    @classmethod
    def _validate_frame(cls, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        missing = [
            column
            for column in cls.REQUIRED_COLUMNS
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {', '.join(missing)}"
            )

        if data.empty:
            raise ValueError("data must not be empty.")

        for column in cls.REQUIRED_COLUMNS:
            if not pd.api.types.is_numeric_dtype(data[column]):
                raise TypeError(f"{column} must be numeric.")

        if data[list(cls.REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError("Required columns must not contain NaN values.")

        if (data["close"] <= 0).any():
            raise ValueError("close values must be positive.")

        if (data["open_interest"] < 0).any():
            raise ValueError("open_interest values must be non-negative.")

    @staticmethod
    def _safe_zscore(series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std(ddof=0)

        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(0.0, index=series.index)

        return (series - mean) / std

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build causal price/OI divergence features.

        Returned columns:
            price_change
            open_interest_change
            price_zscore
            open_interest_zscore
            divergence_score
            divergence_strength
            divergence_confidence
            divergence_signal
        """
        self._validate_frame(data)

        result = data.copy()

        result["price_change"] = (
            result["close"]
            .pct_change(self.lookback)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        result["open_interest_change"] = (
            result["open_interest"]
            .pct_change(self.lookback)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        price_mean = result["price_change"].rolling(
            self.zscore_window,
            min_periods=2,
        ).mean()

        price_std = result["price_change"].rolling(
            self.zscore_window,
            min_periods=2,
        ).std(ddof=0)

        oi_mean = result["open_interest_change"].rolling(
            self.zscore_window,
            min_periods=2,
        ).mean()

        oi_std = result["open_interest_change"].rolling(
            self.zscore_window,
            min_periods=2,
        ).std(ddof=0)

        result["price_zscore"] = (
            (result["price_change"] - price_mean)
            / price_std.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        result["open_interest_zscore"] = (
            (result["open_interest_change"] - oi_mean)
            / oi_std.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        result["divergence_score"] = (
            result["open_interest_zscore"]
            - result["price_zscore"]
        )

        result["divergence_strength"] = (
            result["divergence_score"].abs()
        )

        result["divergence_confidence"] = (
            result["divergence_strength"]
            / (
                1.0
                + result["price_zscore"].abs()
                + result["open_interest_zscore"].abs()
            )
        ).clip(0.0, 1.0)

        bullish = (
            (result["price_change"] < 0)
            & (result["open_interest_change"] > 0)
            & (
                result["divergence_strength"]
                >= self.min_divergence
            )
            & (
                result["divergence_confidence"]
                >= self.min_confidence
            )
        )

        bearish = (
            (result["price_change"] > 0)
            & (result["open_interest_change"] < 0)
            & (
                result["divergence_strength"]
                >= self.min_divergence
            )
            & (
                result["divergence_confidence"]
                >= self.min_confidence
            )
        )

        result["divergence_signal"] = np.select(
            [bullish, bearish],
            ["BULLISH", "BEARISH"],
            default="NEUTRAL",
        )

        return result

    def evaluate(
        self,
        data: pd.DataFrame,
    ) -> OpenInterestDivergenceResult:
        transformed = self.transform(data)
        row = transformed.iloc[-1]

        signal = str(row["divergence_signal"])

        return OpenInterestDivergenceResult(
            signal=signal,
            strength=float(row["divergence_strength"]),
            price_change=float(row["price_change"]),
            open_interest_change=float(row["open_interest_change"]),
            price_zscore=float(row["price_zscore"]),
            open_interest_zscore=float(row["open_interest_zscore"]),
            divergence_score=float(row["divergence_score"]),
            confidence=float(row["divergence_confidence"]),
        )

    def signal(
        self,
        data: pd.DataFrame,
    ) -> str:
        return self.evaluate(data).signal

    def compare(
        self,
        data: pd.DataFrame,
        expected_signal: str,
    ) -> bool:
        if not isinstance(expected_signal, str):
            raise TypeError("expected_signal must be a string.")

        return self.signal(data) == expected_signal.upper()

    def calculate_divergence(
        self,
        price: pd.Series,
        open_interest: pd.Series,
    ) -> pd.DataFrame:
        if not isinstance(price, pd.Series):
            raise TypeError("price must be a pandas Series.")

        if not isinstance(open_interest, pd.Series):
            raise TypeError("open_interest must be a pandas Series.")

        if len(price) != len(open_interest):
            raise ValueError(
                "price and open_interest must have equal length."
            )

        frame = pd.DataFrame(
            {
                "close": price,
                "open_interest": open_interest,
            }
        )

        return self.transform(frame)

    @staticmethod
    def summarize(
        data: pd.DataFrame,
    ) -> Mapping[str, float | str]:
        required = {
            "divergence_strength",
            "divergence_confidence",
            "divergence_signal",
        }

        missing = required.difference(data.columns)

        if missing:
            raise ValueError(
                f"Missing divergence columns: {', '.join(sorted(missing))}"
            )

        if data.empty:
            raise ValueError("data must not be empty.")

        latest = data.iloc[-1]

        return {
            "signal": str(latest["divergence_signal"]),
            "strength": float(latest["divergence_strength"]),
            "confidence": float(latest["divergence_confidence"]),
        }