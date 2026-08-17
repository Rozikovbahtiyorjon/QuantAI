from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FuturesDataConfig:
    open_interest_window: int = 20
    funding_window: int = 20
    basis_window: int = 20
    volume_window: int = 20
    min_history: int = 2

    def __post_init__(self) -> None:
        for name in (
            "open_interest_window",
            "funding_window",
            "basis_window",
            "volume_window",
            "min_history",
        ):
            value = getattr(self, name)

            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"{name} must be a positive integer."
                )


class FuturesDataEngine:
    REQUIRED_COLUMNS = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
        "funding_rate",
    }

    OPTIONAL_COLUMNS = {
        "mark_price",
        "index_price",
        "taker_buy_volume",
    }

    OUTPUT_COLUMNS = (
        "futures_return",
        "open_interest_change",
        "open_interest_change_pct",
        "open_interest_volume_ratio",
        "funding_rate_bps",
        "funding_rate_zscore",
        "basis",
        "basis_bps",
        "basis_zscore",
        "taker_buy_volume_ratio",
    )

    def __init__(
        self,
        config: FuturesDataConfig | None = None,
    ) -> None:
        self.config = config or FuturesDataConfig()

    @classmethod
    def validate_input(
        cls,
        df: pd.DataFrame,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        missing = cls.REQUIRED_COLUMNS.difference(
            df.columns
        )

        if missing:
            raise ValueError(
                "Missing required futures columns: "
                f"{sorted(missing)}"
            )

        if df.empty:
            raise ValueError(
                "Futures DataFrame must not be empty."
            )

        if df["timestamp"].isna().any():
            raise ValueError(
                "timestamp must not contain null values."
            )

        if df["timestamp"].duplicated().any():
            raise ValueError(
                "timestamp must contain unique observations."
            )

        numeric_columns = (
            set(cls.REQUIRED_COLUMNS) - {"timestamp"}
        ).union(
            cls.OPTIONAL_COLUMNS.intersection(
                df.columns
            )
        )

        for column in numeric_columns:
            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):
                raise TypeError(
                    f"{column} must be numeric."
                )

        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        ):
            values = df[column].to_numpy(
                dtype=float
            )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{column} contains non-finite values."
                )

        if (df["close"] <= 0).any():
            raise ValueError(
                "close must be strictly positive."
            )

        if (df["volume"] < 0).any():
            raise ValueError(
                "volume must be non-negative."
            )

        if (df["open_interest"] < 0).any():
            raise ValueError(
                "open_interest must be non-negative."
            )

        if (df["high"] < df["low"]).any():
            raise ValueError(
                "high must be greater than or equal to low."
            )

    @staticmethod
    def _sorted_copy(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        result = df.copy()

        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="raise",
        )

        return (
            result
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    @staticmethod
    def _safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series:
        denominator = denominator.replace(
            0.0,
            np.nan,
        )

        return (
            numerator
            .divide(denominator)
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    @staticmethod
    def _zscore(
        series: pd.Series,
        window: int,
    ) -> pd.Series:
        mean = series.rolling(
            window=window,
            min_periods=window,
        ).mean()

        std = series.rolling(
            window=window,
            min_periods=window,
        ).std(ddof=0)

        return series.subtract(mean).divide(
            std.replace(0.0, np.nan)
        )

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input(df)

        result = self._sorted_copy(df)

        result["futures_return"] = (
            result["close"].pct_change()
        )

        result["open_interest_change"] = (
            result["open_interest"].diff()
        )

        result["open_interest_change_pct"] = (
            result["open_interest"].pct_change()
        )

        volume_mean = (
            result["volume"]
            .rolling(
                self.config.volume_window,
                min_periods=self.config.min_history,
            )
            .mean()
        )

        result["open_interest_volume_ratio"] = (
            self._safe_divide(
                result["open_interest"],
                volume_mean,
            )
        )

        result["funding_rate_bps"] = (
            result["funding_rate"] * 10_000.0
        )

        result["funding_rate_zscore"] = (
            self._zscore(
                result["funding_rate"],
                self.config.funding_window,
            )
        )

        if {
            "mark_price",
            "index_price",
        }.issubset(result.columns):
            index_price = result[
                "index_price"
            ].replace(
                0.0,
                np.nan,
            )

            result["basis"] = (
                result["mark_price"]
                .subtract(index_price)
            )

            result["basis_bps"] = (
                self._safe_divide(
                    result["basis"],
                    index_price,
                )
                * 10_000.0
            )

            result["basis_zscore"] = (
                self._zscore(
                    result["basis_bps"],
                    self.config.basis_window,
                )
            )

        else:
            result["basis"] = np.nan
            result["basis_bps"] = np.nan
            result["basis_zscore"] = np.nan

        if "taker_buy_volume" in result.columns:
            result["taker_buy_volume_ratio"] = (
                self._safe_divide(
                    result["taker_buy_volume"],
                    result["volume"],
                )
            )

        else:
            result["taker_buy_volume_ratio"] = np.nan

        return result

    def latest(
        self,
        df: pd.DataFrame,
    ) -> dict[str, float | pd.Timestamp]:
        transformed = self.transform(df)
        row = transformed.iloc[-1]

        output: dict[
            str,
            float | pd.Timestamp,
        ] = {
            "timestamp": row["timestamp"],
        }

        for column in self.OUTPUT_COLUMNS:
            value = row[column]

            output[column] = (
                float(value)
                if pd.notna(value)
                else float("nan")
            )

        return output

    def feature_columns(
        self,
    ) -> tuple[str, ...]:
        return self.OUTPUT_COLUMNS


def build_futures_features(
    df: pd.DataFrame,
    config: FuturesDataConfig | None = None,
) -> pd.DataFrame:
    return FuturesDataEngine(config).transform(df)


__all__ = [
    "FuturesDataConfig",
    "FuturesDataEngine",
    "build_futures_features",
]