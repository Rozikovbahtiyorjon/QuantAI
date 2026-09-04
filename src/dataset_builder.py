"""
====================================================
QuantAI Professional v5.0
Dataset Builder v2.1
====================================================

Назначение

Автоматическое построение обучающего датасета
для Machine Learning.

Pipeline:

    OHLCV
       ↓
    Indicators
       ↓
    Feature Engine
       ↓
    Features
       ↓
    Future Target
       ↓
    Clean Dataset

Важно:

DatasetBuilder ожидает исходный OHLCV DataFrame,
но автоматически рассчитывает необходимые индикаторы.

На выходе:

    DataFrame
    готовый для MLEngine.
====================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from src.indicators import add_indicators
from src.feature_engine import build_features


# ====================================================
# DATASET CONFIG
# ====================================================

@dataclass
class DatasetConfig:
    """
    Configuration for dataset generation.

    Supports two labeling modes:
      - "simple" (legacy): future close vs current close ± target_profit
      - "triple_barrier" (Phase 1): path-dependent, cost-aware
    """

    # Number of candles into the future
    # used for target generation.
    future_bars: int = 5

    # Minimum future return required
    # to classify BUY / SELL (simple mode).
    target_profit: float = 0.002

    # Labeling mode: "simple" | "triple_barrier"
    label_method: str = "triple_barrier"

    # Triple-barrier params (used when label_method == "triple_barrier")
    tb_pt: float = 0.012
    tb_sl: float = 0.008
    tb_use_atr: bool = True
    tb_atr_pt_mult: float = 3.0
    tb_atr_sl_mult: float = 1.5
    tb_min_net_return: float | None = None  # None → 2*(commission+slippage)

    # Minimum history required before
    # Feature Engine starts generating features.
    warmup_bars: int = 200

    # Remove rows containing NaN / infinite values.
    drop_nan: bool = True

    # Automatically calculate technical indicators.
    calculate_indicators: bool = True

    # Feature Store auto-materialization
    feature_store_enabled: bool = False
    feature_store_view: str = "default"
    feature_store_root: str = "data/feature_store"


# ====================================================
# DATASET BUILDER
# ====================================================

class DatasetBuilder:
    """
    Builds an ML-ready dataset.

    Input:

        OHLCV DataFrame

    Internal pipeline:

        OHLCV
            ↓
        Indicators
            ↓
        Feature Engine
            ↓
        Future Target

    Output:

        pandas.DataFrame
    """

    def __init__(
        self,
        config: DatasetConfig | None = None,
        feature_store=None,
    ) -> None:

        self.config = (
            config
            or DatasetConfig()
        )

        self.dataset: List[dict] = []
        self._feature_store = feature_store
        self._last_materialized_version: int | None = None

    # ====================================================
    # RESET
    # ====================================================

    def reset(self) -> None:
        """
        Clear previously generated dataset.
        """

        self.dataset = []

    # ====================================================
    # VALIDATE INPUT
    # ====================================================

    @staticmethod
    def validate_data(
        df: pd.DataFrame,
    ) -> None:
        """
        Validate incoming market data.
        """

        if not isinstance(
            df,
            pd.DataFrame,
        ):

            raise TypeError(
                "DatasetBuilder requires "
                "a pandas DataFrame."
            )

        if df.empty:

            raise ValueError(
                "DatasetBuilder received "
                "an empty DataFrame."
            )

        required_columns = {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:

            raise ValueError(
                "DatasetBuilder is missing "
                f"required OHLCV columns: "
                f"{sorted(missing)}"
            )

    # ====================================================
    # PREPARE DATA
    # ====================================================

    def prepare_data(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare market data.

        Indicators are calculated once on the
        complete historical DataFrame.

        This is important because FeatureEngine
        expects columns such as:

            ema_fast
            ema_slow
            ema_trend
            atr
            volume_sma20

        and other technical indicators.
        """

        self.validate_data(df)

        data = df.copy()

        # ------------------------------------------------
        # Clean index
        # ------------------------------------------------

        data = (
            data
            .reset_index(drop=False)
        )

        # ------------------------------------------------
        # Calculate indicators
        # ------------------------------------------------

        if self.config.calculate_indicators:

            data = add_indicators(
                data
            )

        # ------------------------------------------------
        # Replace infinite values
        # ------------------------------------------------

        data = data.replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )

        return data

    # ====================================================
    # BUILD FEATURES
    # ====================================================

    def build_features_dataset(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Build Feature Engine output.

        Indicators must already exist in df.

        Feature generation starts after warm-up
        period so EMA 200 and other indicators
        have enough historical information.
        """

        self.reset()

        warmup = (
            self.config.warmup_bars
        )

        future = (
            self.config.future_bars
        )

        last_index = (
            len(df)
            - future
        )

        if last_index <= warmup:

            raise ValueError(
                "Not enough candles to build "
                "dataset. "
                f"Received={len(df)}, "
                f"warmup={warmup}, "
                f"future_bars={future}."
            )

        # ------------------------------------------------
        # Generate features
        # ------------------------------------------------

        for i in range(
            warmup,
            last_index,
        ):

            history = (
                df.iloc[: i + 1]
            )

            try:

                features = (
                    build_features(
                        history
                    )
                )

            except KeyError as exc:

                raise ValueError(
                    "FeatureEngine is missing "
                    f"required indicator column: "
                    f"{exc}. "
                    "Make sure add_indicators() "
                    "produces all columns required "
                    "by feature_engine.py."
                ) from exc

            if not isinstance(
                features,
                dict,
            ):

                raise TypeError(
                    "build_features() must "
                    "return a dictionary."
                )

            row = dict(
                features
            )

            # Preserve source candle index.
            row["index"] = i

            self.dataset.append(
                row
            )

    # ====================================================
    # TARGET GENERATOR
    # ====================================================

    def generate_targets(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Generate supervised-learning target.

        Modes:
          simple: future return >= target_profit → BUY etc. (legacy)
          triple_barrier: path-dependent, cost-aware (Phase 1)
                          upper/lower hit first wins, vertical → HOLD
        """
        if self.config.label_method == "triple_barrier":
            self._generate_triple_barrier_targets(df)
            return

        # --- simple legacy ---
        future = self.config.future_bars
        target_profit = self.config.target_profit

        for row in self.dataset:
            i = int(row["index"])
            current_close = float(df.iloc[i]["close"])
            future_close = float(df.iloc[i + future]["close"])
            if current_close <= 0:
                row["target"] = 0
                row["future_return"] = 0.0
                continue
            change = (future_close - current_close) / current_close
            if change >= target_profit:
                target = 1
            elif change <= -target_profit:
                target = -1
            else:
                target = 0
            row["target"] = target
            row["future_return"] = float(change)

    def _generate_triple_barrier_targets(self, df: pd.DataFrame) -> None:
        """Triple-barrier: first of upper/lower/vertical wins, net-of-cost filtered."""
        from src.labeling import TripleBarrierConfig, triple_barrier_label

        cfg = TripleBarrierConfig(
            pt=float(self.config.tb_pt),
            sl=float(self.config.tb_sl),
            max_holding_bars=int(self.config.future_bars),
            use_atr=bool(self.config.tb_use_atr),
            atr_pt_mult=float(self.config.tb_atr_pt_mult),
            atr_sl_mult=float(self.config.tb_atr_sl_mult),
            min_net_return=self.config.tb_min_net_return,
        )
        for row in self.dataset:
            i = int(row["index"])
            r = triple_barrier_label(df, i, cfg)
            row["target"] = int(r["target"])
            row["future_return"] = float(r["ret"])
            # diagnostics for analysis (kept, not used as feature)
            row["tb_barrier"] = r["barrier"]
            row["tb_t1"] = int(r["t1"])

    # ====================================================
    # CLEAN DATASET
    # ====================================================

    def clean_dataset(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Clean final ML dataset.
        """

        result = dataset.copy()

        # ------------------------------------------------
        # Replace infinities
        # ------------------------------------------------

        result = result.replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )

        # ------------------------------------------------
        # Drop NaN
        # ------------------------------------------------

        if self.config.drop_nan:

            result = (
                result
                .dropna()
                .reset_index(
                    drop=True
                )
            )

        return result

    # ====================================================
    # BUILD
    # ====================================================

    def build(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Complete dataset-building pipeline.
        """

        # ------------------------------------------------
        # 1. Prepare OHLCV + indicators
        # ------------------------------------------------

        data = (
            self.prepare_data(
                df
            )
        )

        # ------------------------------------------------
        # 2. Build Feature Engine features
        # ------------------------------------------------

        self.build_features_dataset(
            data
        )

        # ------------------------------------------------
        # 3. Generate future targets
        # ------------------------------------------------

        self.generate_targets(
            data
        )

        # ------------------------------------------------
        # 4. Convert to DataFrame
        # ------------------------------------------------

        dataset = pd.DataFrame(
            self.dataset
        )

        # ------------------------------------------------
        # 5. Clean dataset
        # ------------------------------------------------

        dataset = (
            self.clean_dataset(
                dataset
            )
        )

        # ------------------------------------------------
        # 6. Train-tail drop (prevent look-ahead bias)
        # ------------------------------------------------
        # Drop the last `future_bars` rows since their labels
        # depend on future data that would not be available at prediction time
        if len(dataset) > self.config.future_bars:
            dataset = dataset.iloc[:-self.config.future_bars].copy()

        # ------------------------------------------------
        # 7. Feature Store auto-materialization
        # ------------------------------------------------
        if self.config.feature_store_enabled:
            self._materialize_to_store(dataset, df)

        return dataset

    def _materialize_to_store(
        self,
        dataset: pd.DataFrame,
        source_df: pd.DataFrame,
    ) -> None:
        """Persist dataset to Feature Store with lineage."""
        try:
            from src.feature_store import FeatureStore

            store = self._feature_store
            if store is None:
                store = FeatureStore(self.config.feature_store_root)

            lineage = {
                "source_rows": len(source_df),
                "future_bars": self.config.future_bars,
                "target_profit": self.config.target_profit,
                "warmup_bars": self.config.warmup_bars,
                "dataset_rows": len(dataset),
                "columns": list(dataset.columns),
            }
            result = store.materialize(
                self.config.feature_store_view, dataset, lineage=lineage
            )
            self._last_materialized_version = result["version"]

            # Log drift vs previous if available
            if result.get("drift_vs_previous"):
                drifted = sum(
                    1 for r in result["drift_vs_previous"] if r.get("drifted")
                )
                if drifted > 0:
                    print(
                        f"[FeatureStore] v{result['version']} drift detected: "
                        f"{drifted}/{len(result['drift_vs_previous'])} features"
                    )
        except Exception as e:
            print(f"[FeatureStore] materialization failed: {e}")

    # ====================================================
    # STATISTICS
    # ====================================================

    @staticmethod
    def statistics(
        dataset: pd.DataFrame,
    ) -> dict:
        """
        Return dataset statistics.
        """

        if dataset.empty:

            return {
                "rows": 0,
                "columns": 0,
                "buy": 0,
                "sell": 0,
                "hold": 0,
                "buy_percent": 0.0,
                "sell_percent": 0.0,
                "hold_percent": 0.0,
            }

        total = len(
            dataset
        )

        buy = int(
            (
                dataset["target"]
                == 1
            ).sum()
        )

        sell = int(
            (
                dataset["target"]
                == -1
            ).sum()
        )

        hold = int(
            (
                dataset["target"]
                == 0
            ).sum()
        )

        return {
            "rows": total,
            "columns": len(
                dataset.columns
            ),
            "buy": buy,
            "sell": sell,
            "hold": hold,
            "buy_percent": (
                buy / total * 100
            ),
            "sell_percent": (
                sell / total * 100
            ),
            "hold_percent": (
                hold / total * 100
            ),
        }

    # ====================================================
    # PRINT STATISTICS
    # ====================================================

    @staticmethod
    def print_statistics(
        dataset: pd.DataFrame,
    ) -> None:
        """
        Print human-readable dataset statistics.
        """

        stats = (
            DatasetBuilder
            .statistics(
                dataset
            )
        )

        print()

        print(
            "=" * 60
        )

        print(
            "QUANTAI DATASET STATISTICS"
        )

        print(
            "=" * 60
        )

        print(
            f"Rows       : "
            f"{stats['rows']}"
        )

        print(
            f"Columns    : "
            f"{stats['columns']}"
        )

        print(
            "-" * 60
        )

        print(
            f"BUY        : "
            f"{stats['buy']} "
            f"({stats['buy_percent']:.2f}%)"
        )

        print(
            f"SELL       : "
            f"{stats['sell']} "
            f"({stats['sell_percent']:.2f}%)"
        )

        print(
            f"HOLD       : "
            f"{stats['hold']} "
            f"({stats['hold_percent']:.2f}%)"
        )

        print(
            "=" * 60
        )

        print()

    # ====================================================
    # TARGET DISTRIBUTION
    # ====================================================

    @staticmethod
    def target_distribution(
        dataset: pd.DataFrame,
    ) -> pd.Series:
        """
        Return target class distribution.
        """

        if (
            dataset.empty
            or "target"
            not in dataset.columns
        ):

            return pd.Series(
                dtype="int64"
            )

        return (
            dataset["target"]
            .value_counts()
            .sort_index()
        )


# ====================================================
# PUBLIC API
# ====================================================

def build_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convenience function.

    Example:

        dataset = build_dataset(df)
    """

    builder = DatasetBuilder()

    return builder.build(
        df
    )


# ====================================================
# EXPORTS
# ====================================================

__all__ = [
    "DatasetConfig",
    "DatasetBuilder",
    "build_dataset",
]