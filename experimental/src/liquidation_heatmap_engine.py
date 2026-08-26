from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LiquidationHeatmapConfig:
    price_bins: int = 50
    lookback: int = 100
    level_quantile: float = 0.75
    min_event_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.price_bins, int) or self.price_bins < 2:
            raise ValueError("price_bins must be an integer >= 2.")

        if not isinstance(self.lookback, int) or self.lookback < 1:
            raise ValueError("lookback must be a positive integer.")

        if not 0.0 < self.level_quantile <= 1.0:
            raise ValueError(
                "level_quantile must be greater than 0 and <= 1."
            )

        if (
            not isinstance(self.min_event_count, int)
            or self.min_event_count < 1
        ):
            raise ValueError(
                "min_event_count must be a positive integer."
            )


class LiquidationHeatmapEngine:
    REQUIRED_COLUMNS = {
        "timestamp",
        "price",
        "side",
        "quantity",
    }

    OPTIONAL_COLUMNS = {
        "notional",
        "symbol",
    }

    OUTPUT_COLUMNS = (
        "price",
        "long_liquidation_volume",
        "short_liquidation_volume",
        "total_liquidation_volume",
        "long_event_count",
        "short_event_count",
        "total_event_count",
        "liquidation_imbalance",
        "relative_intensity",
        "level_strength",
    )

    def __init__(
        self,
        config: LiquidationHeatmapConfig | None = None,
    ) -> None:
        self.config = config or LiquidationHeatmapConfig()

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
                "Missing required liquidation columns: "
                f"{sorted(missing)}"
            )

        if df.empty:
            raise ValueError(
                "Liquidation DataFrame must not be empty."
            )

        if df["timestamp"].isna().any():
            raise ValueError(
                "timestamp must not contain null values."
            )

        if df["timestamp"].duplicated().any():
            raise ValueError(
                "timestamp must contain unique observations."
            )

        for column in ("price", "quantity"):
            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):
                raise TypeError(
                    f"{column} must be numeric."
                )

            values = df[column].to_numpy(
                dtype=float
            )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{column} contains non-finite values."
                )

        if (df["price"] <= 0).any():
            raise ValueError(
                "price must be strictly positive."
            )

        if (df["quantity"] <= 0).any():
            raise ValueError(
                "quantity must be strictly positive."
            )

        sides = (
            df["side"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        allowed = {"long", "short"}

        if not sides.isin(allowed).all():
            raise ValueError(
                "side must contain only 'long' or 'short'."
            )

    @staticmethod
    def _prepare(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        result = df.copy()

        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="raise",
        )

        result["side"] = (
            result["side"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        if "notional" in result.columns:
            result["notional"] = pd.to_numeric(
                result["notional"],
                errors="raise",
            )
        else:
            result["notional"] = (
                result["price"]
                * result["quantity"]
            )

        result["notional"] = result[
            "notional"
        ].astype(float)

        if (result["notional"] <= 0).any():
            raise ValueError(
                "notional must be strictly positive."
            )

        return (
            result
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def build_heatmap(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_input(df)

        data = self._prepare(df)

        data = (
            data
            .tail(self.config.lookback)
            .reset_index(drop=True)
        )

        minimum = float(
            data["price"].min()
        )

        maximum = float(
            data["price"].max()
        )

        if minimum == maximum:
            edges = np.linspace(
                minimum * 0.999,
                maximum * 1.001,
                self.config.price_bins + 1,
            )
        else:
            edges = np.linspace(
                minimum,
                maximum,
                self.config.price_bins + 1,
            )

        data["price_bin"] = pd.cut(
            data["price"],
            bins=edges,
            labels=False,
            include_lowest=True,
            duplicates="drop",
        )

        grouped = (
            data.groupby(
                "price_bin",
                observed=True,
            )
            .agg(
                long_liquidation_volume=(
                    "notional",
                    lambda values: float(
                        values[
                            data.loc[
                                values.index,
                                "side",
                            ]
                            == "long"
                        ].sum()
                    ),
                ),
                short_liquidation_volume=(
                    "notional",
                    lambda values: float(
                        values[
                            data.loc[
                                values.index,
                                "side",
                            ]
                            == "short"
                        ].sum()
                    ),
                ),
                long_event_count=(
                    "side",
                    lambda values: int(
                        (values == "long").sum()
                    ),
                ),
                short_event_count=(
                    "side",
                    lambda values: int(
                        (values == "short").sum()
                    ),
                ),
            )
            .reset_index()
        )

        grouped["total_liquidation_volume"] = (
            grouped["long_liquidation_volume"]
            + grouped["short_liquidation_volume"]
        )

        grouped["total_event_count"] = (
            grouped["long_event_count"]
            + grouped["short_event_count"]
        )

        grouped["price"] = grouped[
            "price_bin"
        ].map(
            lambda value: (
                edges[int(value)]
                + edges[int(value) + 1]
            )
            / 2.0
        )

        grouped["liquidation_imbalance"] = (
            grouped["long_liquidation_volume"]
            - grouped["short_liquidation_volume"]
        ).divide(
            grouped[
                "total_liquidation_volume"
            ].replace(
                0.0,
                np.nan,
            )
        )

        maximum_volume = float(
            grouped[
                "total_liquidation_volume"
            ].max()
        )

        if maximum_volume > 0:
            grouped["relative_intensity"] = (
                grouped[
                    "total_liquidation_volume"
                ]
                / maximum_volume
            )
        else:
            grouped["relative_intensity"] = 0.0

        threshold = float(
            grouped[
                "total_liquidation_volume"
            ].quantile(
                self.config.level_quantile
            )
        )

        grouped["level_strength"] = np.where(
            (
                grouped[
                    "total_liquidation_volume"
                ]
                >= threshold
            )
            & (
                grouped["total_event_count"]
                >= self.config.min_event_count
            ),
            grouped["relative_intensity"],
            0.0,
        )

        return (
            grouped[
                list(self.OUTPUT_COLUMNS)
            ]
            .sort_values("price")
            .reset_index(drop=True)
        )

    def levels(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        heatmap = self.build_heatmap(df)

        return (
            heatmap[
                heatmap["level_strength"] > 0.0
            ]
            .copy()
            .reset_index(drop=True)
        )

    def nearest_levels(
        self,
        df: pd.DataFrame,
        current_price: float,
    ) -> dict[str, float | None]:
        if not isinstance(
            current_price,
            (int, float),
        ):
            raise TypeError(
                "current_price must be numeric."
            )

        if not np.isfinite(
            float(current_price)
        ):
            raise ValueError(
                "current_price must be finite."
            )

        if float(current_price) <= 0:
            raise ValueError(
                "current_price must be strictly positive."
            )

        levels = self.levels(df)

        if levels.empty:
            return {
                "nearest_support": None,
                "nearest_resistance": None,
            }

        below = levels[
            levels["price"]
            < float(current_price)
        ]

        above = levels[
            levels["price"]
            > float(current_price)
        ]

        support = (
            float(below.iloc[-1]["price"])
            if not below.empty
            else None
        )

        resistance = (
            float(above.iloc[0]["price"])
            if not above.empty
            else None
        )

        return {
            "nearest_support": support,
            "nearest_resistance": resistance,
        }

    def latest(
        self,
        df: pd.DataFrame,
        current_price: float | None = None,
    ) -> dict[str, float | None]:
        levels = self.levels(df)

        if levels.empty:
            result: dict[
                str,
                float | None,
            ] = {
                "strongest_level": None,
                "strongest_level_strength": None,
            }
        else:
            strongest = levels.loc[
                levels["level_strength"].idxmax()
            ]

            result = {
                "strongest_level": float(
                    strongest["price"]
                ),
                "strongest_level_strength": float(
                    strongest[
                        "level_strength"
                    ]
                ),
            }

        if current_price is not None:
            result.update(
                self.nearest_levels(
                    df,
                    current_price,
                )
            )

        return result


def build_liquidation_heatmap(
    df: pd.DataFrame,
    config: LiquidationHeatmapConfig | None = None,
) -> pd.DataFrame:
    return LiquidationHeatmapEngine(
        config
    ).build_heatmap(df)


__all__ = [
    "LiquidationHeatmapConfig",
    "LiquidationHeatmapEngine",
    "build_liquidation_heatmap",
]