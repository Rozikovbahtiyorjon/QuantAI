from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class DerivativesMarketData:
    symbol: str
    timestamp: int
    price: float
    open_interest: float
    funding_rate: float
    liquidation_volume: float
    long_short_ratio: float
    spot_price: float | None = None

    @property
    def basis(self) -> float | None:
        if self.spot_price is None:
            return None

        return self.price - self.spot_price

    @property
    def basis_percent(self) -> float | None:
        if (
            self.spot_price is None
            or self.spot_price == 0
        ):
            return None

        return (
            self.price / self.spot_price - 1.0
        ) * 100.0


@dataclass(frozen=True)
class DerivativesSignal:
    price_change: float
    open_interest_change: float
    price_oi_divergence: bool
    context: str


class DerivativesMarketDataEngine:
    def __init__(self) -> None:
        self._previous: DerivativesMarketData | None = None

    @property
    def previous(
        self,
    ) -> DerivativesMarketData | None:
        return self._previous

    def update(
        self,
        data: DerivativesMarketData,
    ) -> DerivativesSignal:
        self._validate(data)

        if self._previous is None:
            signal = DerivativesSignal(
                price_change=0.0,
                open_interest_change=0.0,
                price_oi_divergence=False,
                context="BASELINE",
            )

            self._previous = data

            return signal

        if data.symbol != self._previous.symbol:
            raise ValueError(
                "symbol must match the previous market snapshot."
            )

        previous = self._previous

        price_change = self._relative_change(
            previous.price,
            data.price,
        )

        open_interest_change = self._relative_change(
            previous.open_interest,
            data.open_interest,
        )

        divergence = (
            (
                price_change > 0
                and open_interest_change < 0
            )
            or (
                price_change < 0
                and open_interest_change > 0
            )
        )

        if (
            price_change > 0
            and open_interest_change > 0
        ):
            context = "PRICE_UP_OI_UP"

        elif (
            price_change < 0
            and open_interest_change > 0
        ):
            context = "PRICE_DOWN_OI_UP"

        elif (
            price_change > 0
            and open_interest_change < 0
        ):
            context = "PRICE_UP_OI_DOWN"

        elif (
            price_change < 0
            and open_interest_change < 0
        ):
            context = "PRICE_DOWN_OI_DOWN"

        else:
            context = "NEUTRAL"

        self._previous = data

        return DerivativesSignal(
            price_change=price_change,
            open_interest_change=open_interest_change,
            price_oi_divergence=divergence,
            context=context,
        )

    def reset(self) -> None:
        self._previous = None

    @staticmethod
    def _relative_change(
        previous: float,
        current: float,
    ) -> float:
        if previous == 0:
            return 0.0

        return (
            current / previous
        ) - 1.0

    @staticmethod
    def _validate(
        data: DerivativesMarketData,
    ) -> None:
        if not isinstance(
            data,
            DerivativesMarketData,
        ):
            raise TypeError(
                "data must be a DerivativesMarketData instance."
            )

        if (
            not isinstance(data.symbol, str)
            or not data.symbol
        ):
            raise ValueError(
                "symbol must be a non-empty string."
            )

        if data.timestamp < 0:
            raise ValueError(
                "timestamp cannot be negative."
            )

        values = (
            data.price,
            data.open_interest,
            data.funding_rate,
            data.liquidation_volume,
            data.long_short_ratio,
        )

        if data.spot_price is not None:
            values += (data.spot_price,)

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(float(value))
            for value in values
        ):
            raise ValueError(
                "all market metrics must be finite numeric values."
            )

        if data.price <= 0:
            raise ValueError(
                "price must be greater than zero."
            )

        if data.open_interest < 0:
            raise ValueError(
                "open_interest cannot be negative."
            )

        if data.liquidation_volume < 0:
            raise ValueError(
                "liquidation_volume cannot be negative."
            )

        if data.long_short_ratio <= 0:
            raise ValueError(
                "long_short_ratio must be greater than zero."
            )

        if (
            data.spot_price is not None
            and data.spot_price <= 0
        ):
            raise ValueError(
                "spot_price must be greater than zero."
            )


class ExchangeDerivativesMarketData:
    """
    Read-only derivatives market-data adapter.

    Uses an existing ExchangeMarketData instance
    and its CCXT exchange client.

    This class never places or modifies orders.
    """

    def __init__(
        self,
        market_data: Any,
    ) -> None:
        if market_data is None:
            raise TypeError(
                "market_data cannot be None."
            )

        if not hasattr(
            market_data,
            "exchange",
        ):
            raise TypeError(
                "market_data must provide a CCXT exchange."
            )

        self.market_data = market_data
        self.exchange = market_data.exchange

    def fetch(
        self,
        symbol: str,
        spot_symbol: str | None = None,
    ) -> DerivativesMarketData:
        if not isinstance(symbol, str):
            raise TypeError(
                "symbol must be a string."
            )

        symbol = symbol.strip()

        if not symbol:
            raise ValueError(
                "symbol cannot be empty."
            )

        if spot_symbol is None:
            spot_symbol = symbol

        if not isinstance(
            spot_symbol,
            str,
        ):
            raise TypeError(
                "spot_symbol must be a string."
            )

        spot_symbol = spot_symbol.strip()

        if not spot_symbol:
            raise ValueError(
                "spot_symbol cannot be empty."
            )

        ticker = self._fetch_ticker(symbol)

        price = self._extract_price(
            ticker
        )

        timestamp = self._extract_timestamp(
            ticker
        )

        open_interest = (
            self._fetch_open_interest(
                symbol
            )
        )

        funding_rate = (
            self._fetch_funding_rate(
                symbol
            )
        )

        liquidation_volume = (
            self._fetch_liquidation_volume(
                symbol
            )
        )

        long_short_ratio = (
            self._fetch_long_short_ratio(
                symbol
            )
        )

        spot_price = (
            self._fetch_spot_price(
                spot_symbol
            )
        )

        return DerivativesMarketData(
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            open_interest=open_interest,
            funding_rate=funding_rate,
            liquidation_volume=liquidation_volume,
            long_short_ratio=long_short_ratio,
            spot_price=spot_price,
        )

    def _fetch_ticker(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        method = getattr(
            self.exchange,
            "fetch_ticker",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Exchange does not support fetch_ticker."
            )

        result = method(symbol)

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                "Exchange ticker response must be a dictionary."
            )

        return result

    def _fetch_open_interest(
        self,
        symbol: str,
    ) -> float:
        method = getattr(
            self.exchange,
            "fetch_open_interest",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Exchange does not support fetch_open_interest."
            )

        result = method(symbol)

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                "Open interest response must be a dictionary."
            )

        value = result.get(
            "openInterestValue"
        )

        if value is None:
            value = result.get(
                "openInterestAmount"
            )

        if value is None:
            raise ValueError(
                "Open interest response does not contain a supported value."
            )

        return self._finite_float(
            value,
            "open_interest",
        )

    def _fetch_funding_rate(
        self,
        symbol: str,
    ) -> float:
        method = getattr(
            self.exchange,
            "fetch_funding_rate",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Exchange does not support fetch_funding_rate."
            )

        result = method(symbol)

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                "Funding rate response must be a dictionary."
            )

        value = result.get(
            "fundingRate"
        )

        if value is None:
            raise ValueError(
                "Funding rate response does not contain fundingRate."
            )

        return self._finite_float(
            value,
            "funding_rate",
        )

    def _fetch_liquidation_volume(
        self,
        symbol: str,
    ) -> float:
        method = getattr(
            self.exchange,
            "fetch_liquidations",
            None,
        )

        if not callable(method):
            return 0.0

        result = method(
            symbol,
            limit=100,
        )

        if result is None:
            return 0.0

        if not isinstance(
            result,
            list,
        ):
            raise TypeError(
                "Liquidations response must be a list."
            )

        total = 0.0

        for liquidation in result:
            if not isinstance(
                liquidation,
                dict,
            ):
                continue

            amount = liquidation.get(
                "amount"
            )

            price = liquidation.get(
                "price"
            )

            if amount is None:
                continue

            amount_value = self._finite_float(
                amount,
                "liquidation_amount",
            )

            if price is None:
                total += abs(
                    amount_value
                )
                continue

            price_value = self._finite_float(
                price,
                "liquidation_price",
            )

            total += abs(
                amount_value * price_value
            )

        return total

    def _fetch_long_short_ratio(
        self,
        symbol: str,
    ) -> float:
        method = getattr(
            self.exchange,
            "fetch_long_short_ratio",
            None,
        )

        if not callable(method):
            return 1.0

        result = method(symbol)

        if isinstance(
            result,
            list,
        ):
            if not result:
                return 1.0

            result = result[-1]

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                "Long/short ratio response must be a dictionary."
            )

        value = result.get(
            "longShortRatio"
        )

        if value is None:
            value = result.get(
                "longAccount"
            )

        if value is None:
            raise ValueError(
                "Long/short ratio response does not contain a supported value."
            )

        return self._finite_float(
            value,
            "long_short_ratio",
        )

    def _fetch_spot_price(
        self,
        symbol: str,
    ) -> float | None:
        method = getattr(
            self.exchange,
            "fetch_ticker",
            None,
        )

        if not callable(method):
            return None

        result = method(symbol)

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                "Spot ticker response must be a dictionary."
            )

        value = result.get(
            "last"
        )

        if value is None:
            value = result.get(
                "close"
            )

        if value is None:
            return None

        return self._finite_float(
            value,
            "spot_price",
        )

    @staticmethod
    def _extract_price(
        ticker: dict[str, Any],
    ) -> float:
        value = ticker.get("last")

        if value is None:
            value = ticker.get("close")

        if value is None:
            raise ValueError(
                "Ticker response does not contain price."
            )

        return ExchangeDerivativesMarketData._finite_float(
            value,
            "price",
        )

    @staticmethod
    def _extract_timestamp(
        ticker: dict[str, Any],
    ) -> int:
        value = ticker.get("timestamp")

        if value is None:
            raise ValueError(
                "Ticker response does not contain timestamp."
            )

        try:
            timestamp = int(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "Ticker timestamp must be numeric."
            ) from exc

        if timestamp < 0:
            raise ValueError(
                "Ticker timestamp cannot be negative."
            )

        return timestamp

    @staticmethod
    def _finite_float(
        value: Any,
        field_name: str,
    ) -> float:
        try:
            converted = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{field_name} must be numeric."
            ) from exc

        if not isfinite(converted):
            raise ValueError(
                f"{field_name} must be finite."
            )

        return converted


__all__ = [
    "DerivativesMarketData",
    "DerivativesSignal",
    "DerivativesMarketDataEngine",
    "ExchangeDerivativesMarketData",
]