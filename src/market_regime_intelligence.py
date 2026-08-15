from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    symbol: str
    timestamp: int
    trend_score: float
    volatility: float
    volatility_ratio: float
    price_change_percent: float
    regime: str


@dataclass(frozen=True)
class MarketRegimeSignal:
    regime: str
    confidence: float
    trend_score: float
    volatility: float
    volatility_ratio: float
    price_change_percent: float
    context: str


class MarketRegimeIntelligenceEngine:
    VALID_REGIMES = (
        "TREND_UP",
        "TREND_DOWN",
        "RANGE",
        "HIGH_VOLATILITY",
        "LOW_VOLATILITY",
        "SHOCK",
        "RECOVERY",
    )

    def __init__(
        self,
        trend_threshold: float = 0.5,
        high_volatility_ratio: float = 1.5,
        low_volatility_ratio: float = 0.5,
        shock_return_percent: float = 3.0,
        recovery_return_percent: float = 1.0,
    ) -> None:
        self._validate_non_negative(
            trend_threshold,
            "trend_threshold",
        )

        if trend_threshold > 1.0:
            raise ValueError(
                "trend_threshold must be between 0 and 1."
            )

        self._validate_positive(
            high_volatility_ratio,
            "high_volatility_ratio",
        )

        self._validate_positive(
            low_volatility_ratio,
            "low_volatility_ratio",
        )

        if low_volatility_ratio >= high_volatility_ratio:
            raise ValueError(
                "low_volatility_ratio must be lower "
                "than high_volatility_ratio."
            )

        self._validate_positive(
            shock_return_percent,
            "shock_return_percent",
        )

        self._validate_positive(
            recovery_return_percent,
            "recovery_return_percent",
        )

        self.trend_threshold = float(
            trend_threshold
        )

        self.high_volatility_ratio = float(
            high_volatility_ratio
        )

        self.low_volatility_ratio = float(
            low_volatility_ratio
        )

        self.shock_return_percent = float(
            shock_return_percent
        )

        self.recovery_return_percent = float(
            recovery_return_percent
        )

        self._previous: (
            MarketRegimeSnapshot | None
        ) = None

    @property
    def previous(
        self,
    ) -> MarketRegimeSnapshot | None:
        return self._previous

    def classify(
        self,
        symbol: str,
        timestamp: int,
        closes: Sequence[float],
        baseline_volatility: float | None = None,
    ) -> MarketRegimeSignal:
        self._validate_symbol(symbol)
        self._validate_timestamp(timestamp)

        prices = self._validate_prices(
            closes
        )

        if len(prices) < 3:
            raise ValueError(
                "closes must contain at least three prices."
            )

        price_change_percent = (
            (prices[-1] - prices[0])
            / prices[0]
            * 100.0
        )

        returns = [
            (
                prices[index]
                - prices[index - 1]
            )
            / prices[index - 1]
            for index in range(1, len(prices))
        ]

        volatility = self._standard_deviation(
            returns
        )

        if baseline_volatility is None:
            baseline = volatility

        else:
            self._validate_positive(
                baseline_volatility,
                "baseline_volatility",
            )

            baseline = float(
                baseline_volatility
            )

        volatility_ratio = (
            volatility / baseline
            if baseline > 0
            else 1.0
        )

        trend_score = self._trend_score(
            prices
        )

        regime = self._classify_regime(
            trend_score=trend_score,
            volatility_ratio=volatility_ratio,
            price_change_percent=price_change_percent,
            prices=prices,
            previous_regime=(
                self._previous.regime
                if self._previous is not None
                else None
            ),
        )

        confidence = self._confidence(
            trend_score=trend_score,
            volatility_ratio=volatility_ratio,
            regime=regime,
            price_change_percent=price_change_percent,
        )

        context = self._context(
            regime
        )

        snapshot = MarketRegimeSnapshot(
            symbol=symbol.strip(),
            timestamp=timestamp,
            trend_score=trend_score,
            volatility=volatility,
            volatility_ratio=volatility_ratio,
            price_change_percent=price_change_percent,
            regime=regime,
        )

        self._previous = snapshot

        return MarketRegimeSignal(
            regime=regime,
            confidence=confidence,
            trend_score=trend_score,
            volatility=volatility,
            volatility_ratio=volatility_ratio,
            price_change_percent=price_change_percent,
            context=context,
        )

    def reset(self) -> None:
        self._previous = None

    def _classify_regime(
        self,
        trend_score: float,
        volatility_ratio: float,
        price_change_percent: float,
        prices: Sequence[float],
        previous_regime: str | None,
    ) -> str:
        period_returns_percent = [
            abs(
                (
                    prices[index]
                    - prices[index - 1]
                )
                / prices[index - 1]
                * 100.0
            )
            for index in range(1, len(prices))
        ]

        max_period_return = max(
            period_returns_percent
        )

        if (
            max_period_return
            >= self.shock_return_percent
        ):
            return "SHOCK"

        if (
            previous_regime == "SHOCK"
            and abs(price_change_percent)
            >= self.recovery_return_percent
        ):
            return "RECOVERY"

        if (
            volatility_ratio
            >= self.high_volatility_ratio
        ):
            return "HIGH_VOLATILITY"

        if (
            trend_score
            >= self.trend_threshold
        ):
            return "TREND_UP"

        if (
            trend_score
            <= -self.trend_threshold
        ):
            return "TREND_DOWN"

        if (
            volatility_ratio
            <= self.low_volatility_ratio
        ):
            return "LOW_VOLATILITY"

        return "RANGE"

    @staticmethod
    def _trend_score(
        prices: Sequence[float],
    ) -> float:
        first = float(
            prices[0]
        )

        last = float(
            prices[-1]
        )

        if first <= 0:
            raise ValueError(
                "prices must be greater than zero."
            )

        raw = (
            last - first
        ) / first

        if len(prices) == 1:
            return 0.0

        return max(
            -1.0,
            min(
                1.0,
                raw * 10.0,
            ),
        )

    @staticmethod
    def _confidence(
        trend_score: float,
        volatility_ratio: float,
        regime: str,
        price_change_percent: float,
    ) -> float:
        if regime in {
            "TREND_UP",
            "TREND_DOWN",
        }:
            confidence = abs(
                trend_score
            )

        elif regime == "RANGE":
            confidence = (
                1.0
                - min(
                    abs(trend_score),
                    1.0,
                )
            )

        elif regime in {
            "HIGH_VOLATILITY",
            "LOW_VOLATILITY",
        }:
            confidence = min(
                abs(
                    volatility_ratio - 1.0
                ),
                1.0,
            )

        else:
            confidence = min(
                abs(
                    price_change_percent
                )
                / 10.0,
                1.0,
            )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    @staticmethod
    def _context(
        regime: str,
    ) -> str:
        return {
            "TREND_UP": "BULLISH_TREND",
            "TREND_DOWN": "BEARISH_TREND",
            "RANGE": "SIDEWAYS_MARKET",
            "HIGH_VOLATILITY": "ELEVATED_VOLATILITY",
            "LOW_VOLATILITY": "COMPRESSED_VOLATILITY",
            "SHOCK": "MARKET_SHOCK",
            "RECOVERY": "POST_SHOCK_RECOVERY",
        }[regime]

    @staticmethod
    def _standard_deviation(
        values: Sequence[float],
    ) -> float:
        if not values:
            return 0.0

        mean = (
            sum(values)
            / len(values)
        )

        variance = (
            sum(
                (
                    value - mean
                ) ** 2
                for value in values
            )
            / len(values)
        )

        return variance ** 0.5

    @staticmethod
    def _validate_prices(
        closes: Sequence[float],
    ) -> tuple[float, ...]:
        if isinstance(
            closes,
            (str, bytes),
        ):
            raise TypeError(
                "closes must be a sequence "
                "of numeric prices."
            )

        try:
            prices = tuple(
                float(value)
                for value in closes
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                "closes must contain numeric prices."
            ) from exc

        if not prices:
            raise ValueError(
                "closes cannot be empty."
            )

        for price in prices:
            if (
                not isfinite(price)
                or price <= 0
            ):
                raise ValueError(
                    "all closing prices must be "
                    "finite and greater than zero."
                )

        return prices

    @staticmethod
    def _validate_symbol(
        symbol: str,
    ) -> None:
        if not isinstance(
            symbol,
            str,
        ):
            raise TypeError(
                "symbol must be a string."
            )

        if not symbol.strip():
            raise ValueError(
                "symbol cannot be empty."
            )

    @staticmethod
    def _validate_timestamp(
        timestamp: int,
    ) -> None:
        if (
            not isinstance(
                timestamp,
                int,
            )
            or isinstance(
                timestamp,
                bool,
            )
            or timestamp < 0
        ):
            raise ValueError(
                "timestamp must be a "
                "non-negative integer."
            )

    @staticmethod
    def _validate_positive(
        value: float,
        field_name: str,
    ) -> None:
        if (
            not isinstance(
                value,
                (int, float),
            )
            or isinstance(
                value,
                bool,
            )
            or not isfinite(
                float(value)
            )
            or value <= 0
        ):
            raise ValueError(
                f"{field_name} must be finite "
                "and greater than zero."
            )

    @staticmethod
    def _validate_non_negative(
        value: float,
        field_name: str,
    ) -> None:
        if (
            not isinstance(
                value,
                (int, float),
            )
            or isinstance(
                value,
                bool,
            )
            or not isfinite(
                float(value)
            )
            or value < 0
        ):
            raise ValueError(
                f"{field_name} must be finite "
                "and non-negative."
            )


__all__ = [
    "MarketRegimeSnapshot",
    "MarketRegimeSignal",
    "MarketRegimeIntelligenceEngine",
]