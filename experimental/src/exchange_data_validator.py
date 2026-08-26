"""
QuantAI Professional v5
Exchange Data Validator

Validates market data received from an exchange.

This module does NOT:
- connect to an exchange;
- execute orders;
- generate trading signals;
- calculate indicators;
- train ML models;
- modify Strategy;
- modify Trade Engine;
- modify Paper Trading;
- modify Backtest;
- modify Walk-Forward.

It only validates normalized OHLCV data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


# ============================================================
# VALIDATION RESULT
# ============================================================

@dataclass
class ExchangeDataValidationResult:
    """
    Result of exchange market-data validation.
    """

    valid: bool

    errors: List[str] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    rows_checked: int = 0

    duplicate_timestamps: int = 0

    missing_timestamps: int = 0


# ============================================================
# VALIDATOR
# ============================================================

class ExchangeDataValidator:
    """
    Validate normalized exchange OHLCV data.
    """

    def __init__(
        self,
        timeframe_minutes: int = 15,
    ) -> None:

        if timeframe_minutes <= 0:
            raise ValueError(
                "timeframe_minutes must be greater than zero."
            )

        self.timeframe_minutes = int(
            timeframe_minutes
        )

    # ========================================================
    # VALIDATE
    # ========================================================

    def validate(
        self,
        df: pd.DataFrame,
    ) -> ExchangeDataValidationResult:
        """
        Run all market-data validation checks.
        """

        errors: List[str] = []

        warnings: List[str] = []

        duplicate_timestamps = 0

        missing_timestamps = 0

        # ----------------------------------------------------
        # DATAFRAME TYPE
        # ----------------------------------------------------

        if not isinstance(
            df,
            pd.DataFrame,
        ):

            return ExchangeDataValidationResult(
                valid=False,
                errors=[
                    "df must be a pandas DataFrame."
                ],
            )

        rows_checked = len(df)

        # ----------------------------------------------------
        # EMPTY DATAFRAME
        # ----------------------------------------------------

        if df.empty:

            return ExchangeDataValidationResult(
                valid=False,
                errors=[
                    "DataFrame cannot be empty."
                ],
                rows_checked=0,
            )

        # ----------------------------------------------------
        # REQUIRED COLUMNS
        # ----------------------------------------------------

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:

            errors.append(
                "Missing required columns: "
                + ", ".join(
                    missing_columns
                )
            )

        if errors:

            return ExchangeDataValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                rows_checked=rows_checked,
            )

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamps = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )

        invalid_timestamps = int(
            timestamps.isna().sum()
        )

        if invalid_timestamps > 0:

            errors.append(
                f"Invalid timestamps: "
                f"{invalid_timestamps}."
            )

        # ----------------------------------------------------
        # DUPLICATES
        # ----------------------------------------------------

        duplicate_timestamps = int(
            timestamps.duplicated(
                keep=False
            ).sum()
        )

        if duplicate_timestamps > 0:

            errors.append(
                "Duplicate timestamps detected: "
                f"{duplicate_timestamps}."
            )

        # ----------------------------------------------------
        # NUMERIC COLUMNS
        # ----------------------------------------------------

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in numeric_columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            invalid_count = int(
                values.isna().sum()
            )

            if invalid_count > 0:

                errors.append(
                    f"Column '{column}' "
                    f"contains {invalid_count} "
                    "invalid numeric values."
                )

        # ----------------------------------------------------
        # POSITIVE PRICES
        # ----------------------------------------------------

        numeric_df = df.copy()

        for column in numeric_columns:

            numeric_df[column] = pd.to_numeric(
                numeric_df[column],
                errors="coerce",
            )

        invalid_price_rows = (
            (
                numeric_df[
                    [
                        "open",
                        "high",
                        "low",
                        "close",
                    ]
                ]
                <= 0
            )
            .any(axis=1)
            .sum()
        )

        if invalid_price_rows > 0:

            errors.append(
                "Non-positive OHLC prices detected: "
                f"{int(invalid_price_rows)} rows."
            )

        # ----------------------------------------------------
        # HIGH / LOW CONSISTENCY
        # ----------------------------------------------------

        invalid_high_low = (
            (
                numeric_df["high"]
                < numeric_df["low"]
            )
            .fillna(False)
            .sum()
        )

        if invalid_high_low > 0:

            errors.append(
                "High price is lower than low price "
                f"in {int(invalid_high_low)} rows."
            )

        # ----------------------------------------------------
        # OHLC RANGE CONSISTENCY
        # ----------------------------------------------------

        invalid_open_range = (
            (
                (numeric_df["open"] > numeric_df["high"])
                | (numeric_df["open"] < numeric_df["low"])
            )
            .fillna(False)
            .sum()
        )

        if invalid_open_range > 0:

            errors.append(
                "Open price is outside the "
                "high/low range in "
                f"{int(invalid_open_range)} rows."
            )

        invalid_close_range = (
            (
                (numeric_df["close"] > numeric_df["high"])
                | (numeric_df["close"] < numeric_df["low"])
            )
            .fillna(False)
            .sum()
        )

        if invalid_close_range > 0:

            errors.append(
                "Close price is outside the "
                "high/low range in "
                f"{int(invalid_close_range)} rows."
            )

        # ----------------------------------------------------
        # VOLUME
        # ----------------------------------------------------

        negative_volume = int(
            (
                numeric_df["volume"]
                < 0
            )
            .fillna(False)
            .sum()
        )

        if negative_volume > 0:

            errors.append(
                "Negative volume detected: "
                f"{negative_volume} rows."
            )

        # ----------------------------------------------------
        # CHRONOLOGICAL ORDER
        # ----------------------------------------------------

        valid_timestamps = timestamps.dropna()

        if len(valid_timestamps) > 1:

            differences = (
                valid_timestamps
                .sort_values()
                .diff()
                .dropna()
            )

            expected_delta = pd.Timedelta(
                minutes=self.timeframe_minutes
            )

            missing_intervals = int(
                (
                    differences
                    > expected_delta
                )
                .sum()
            )

            missing_timestamps = missing_intervals

            if missing_intervals > 0:

                warnings.append(
                    "Missing candle intervals detected: "
                    f"{missing_intervals}."
                )

        # ----------------------------------------------------
        # ORDER CHECK
        # ----------------------------------------------------

        if not timestamps.is_monotonic_increasing:

            errors.append(
                "Timestamps are not in chronological order."
            )

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        valid = (
            len(errors) == 0
        )

        return ExchangeDataValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            rows_checked=rows_checked,
            duplicate_timestamps=duplicate_timestamps,
            missing_timestamps=missing_timestamps,
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def validate_exchange_data(
    df: pd.DataFrame,
    timeframe_minutes: int = 15,
) -> ExchangeDataValidationResult:
    """
    Validate exchange OHLCV data.
    """

    validator = ExchangeDataValidator(
        timeframe_minutes=timeframe_minutes
    )

    return validator.validate(
        df
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "REQUIRED_COLUMNS",
    "ExchangeDataValidationResult",
    "ExchangeDataValidator",
    "validate_exchange_data",
]