from __future__ import annotations

from typing import Any, Optional

import ccxt
import pandas as pd


OHLCV_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

DEFAULT_EXCHANGE = "binance"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_LIMIT = 500


class ExchangeMarketData:
    """
    Read-only exchange market-data provider.

    The class uses ccxt and does not perform any trading operations.
    """

    def __init__(
        self,
        exchange_id: str = DEFAULT_EXCHANGE,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        timeout: int = 10000,
    ) -> None:
        if not isinstance(exchange_id, str):
            raise TypeError(
                "exchange_id must be a string."
            )

        exchange_id = exchange_id.strip().lower()

        if not exchange_id:
            raise ValueError(
                "exchange_id cannot be empty."
            )

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        exchange_class = getattr(
            ccxt,
            exchange_id,
            None,
        )

        if exchange_class is None:
            raise ValueError(
                f"Unsupported exchange: {exchange_id}"
            )

        config: dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": int(timeout),
        }

        if api_key is not None:
            config["apiKey"] = str(api_key)

        if secret is not None:
            config["secret"] = str(secret)

        self.exchange_id = exchange_id

        self.exchange = exchange_class(
            config
        )

    @property
    def has_fetch_ohlcv(self) -> bool:
        """
        Return True when the exchange supports OHLCV data.
        """

        return bool(
            self.exchange.has.get(
                "fetchOHLCV",
                False,
            )
        )

    def load_markets(self) -> dict[str, Any]:
        """
        Load exchange market metadata.
        """

        return self.exchange.load_markets()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
        limit: int = DEFAULT_LIMIT,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles from the exchange.

        Parameters
        ----------
        symbol:
            Exchange symbol such as BTC/USDT.

        timeframe:
            Candle timeframe such as 1m, 5m, 15m, 1h.

        limit:
            Maximum number of candles requested.

        since:
            Optional Unix timestamp in milliseconds.

        Returns
        -------
        pandas.DataFrame
            Normalized OHLCV data.
        """

        if not isinstance(
            symbol,
            str,
        ):
            raise TypeError(
                "symbol must be a string."
            )

        symbol = symbol.strip()

        if not symbol:
            raise ValueError(
                "symbol cannot be empty."
            )

        if not isinstance(
            timeframe,
            str,
        ):
            raise TypeError(
                "timeframe must be a string."
            )

        timeframe = timeframe.strip()

        if not timeframe:
            raise ValueError(
                "timeframe cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        if not self.has_fetch_ohlcv:
            raise RuntimeError(
                f"Exchange {self.exchange_id} "
                "does not support fetchOHLCV."
            )

        params: dict[str, Any] = {}

        if since is None:
            raw_data = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=int(limit),
                params=params,
            )

        else:
            if since < 0:
                raise ValueError(
                    "since cannot be negative."
                )

            raw_data = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=int(since),
                limit=int(limit),
                params=params,
            )

        return self._normalize_ohlcv(
            raw_data
        )

    @staticmethod
    def _normalize_ohlcv(
        raw_data: Any,
    ) -> pd.DataFrame:
        """
        Normalize raw exchange OHLCV data.
        """

        if raw_data is None:
            return pd.DataFrame(
                columns=OHLCV_COLUMNS
            )

        if not isinstance(
            raw_data,
            list,
        ):
            raise TypeError(
                "Exchange OHLCV response must be a list."
            )

        if not raw_data:
            return pd.DataFrame(
                columns=OHLCV_COLUMNS
            )

        rows = []

        for row in raw_data:

            if not isinstance(
                row,
                (list, tuple),
            ):
                raise ValueError(
                    "Invalid OHLCV row."
                )

            if len(row) < 6:
                raise ValueError(
                    "OHLCV row must contain "
                    "at least six values."
                )

            rows.append(
                [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                ]
            )

        df = pd.DataFrame(
            rows,
            columns=OHLCV_COLUMNS,
        )

        for column in OHLCV_COLUMNS:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            errors="coerce",
            utc=True,
        )

        df = df.dropna(
            subset=OHLCV_COLUMNS
        ).copy()

        df = df.sort_values(
            "timestamp"
        )

        df = df.drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )

        df = df.reset_index(
            drop=True
        )

        return df

    def fetch_latest(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> pd.Series:
        """
        Fetch the latest available candle.
        """

        df = self.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=1,
        )

        if df.empty:
            raise RuntimeError(
                "Exchange returned no OHLCV data."
            )

        return df.iloc[-1]

    def close(self) -> None:
        """
        Close the exchange connection when supported.
        """

        close_method = getattr(
            self.exchange,
            "close",
            None,
        )

        if callable(close_method):
            close_method()


def fetch_exchange_ohlcv(
    symbol: str,
    exchange_id: str = DEFAULT_EXCHANGE,
    timeframe: str = DEFAULT_TIMEFRAME,
    limit: int = DEFAULT_LIMIT,
    since: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience function for fetching exchange OHLCV data.
    """

    provider = ExchangeMarketData(
        exchange_id=exchange_id
    )

    try:
        return provider.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            since=since,
        )

    finally:
        provider.close()


__all__ = [
    "OHLCV_COLUMNS",
    "DEFAULT_EXCHANGE",
    "DEFAULT_TIMEFRAME",
    "DEFAULT_LIMIT",
    "ExchangeMarketData",
    "fetch_exchange_ohlcv",
]