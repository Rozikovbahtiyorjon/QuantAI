"""
====================================================
QuantAI Professional v5.0
Dataset Builder
====================================================

Назначение

Автоматическое построение датасета
для обучения моделей Machine Learning.

На входе:

    DataFrame OHLCV
    +
    Индикаторы
    +
    Feature Engine

На выходе:

    DataFrame,
    готовый для обучения AI.

====================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from src.feature_engine import build_features


# ====================================================
# DATASET CONFIG
# ====================================================

@dataclass
class DatasetConfig:

    future_bars: int = 5

    target_profit: float = 0.003

    drop_nan: bool = True


# ====================================================
# DATASET BUILDER
# ====================================================

class DatasetBuilder:

    """
    Построение обучающего датасета.

    Каждая строка датасета содержит:

        признаки рынка

        будущую цель (Target)
    """

    def __init__(

        self,

        config: DatasetConfig | None = None,

    ):

        self.config = config or DatasetConfig()

        self.dataset: List[dict] = []

    def reset(self):

        self.dataset = []

            # ====================================================
    # BUILD FEATURES
    # ====================================================

    def build_features_dataset(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Строит признаки для каждой свечи,
        начиная с 200-й, чтобы все индикаторы
        уже были рассчитаны.
        """

        self.reset()

        for i in range(200, len(df) - self.config.future_bars):

            history = df.iloc[: i + 1]

            features = build_features(history)

            row = dict(features)

            row["index"] = i

            self.dataset.append(row)

                # ====================================================
    # TARGET GENERATOR
    # ====================================================

    def generate_targets(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Генерирует целевую переменную (Target)
        для обучения модели.

        Target:

            1  = BUY

            -1 = SELL

            0  = HOLD
        """

        future = self.config.future_bars

        target_profit = self.config.target_profit

        for row in self.dataset:

            i = row["index"]

            current_close = float(df.iloc[i]["close"])

            future_close = float(df.iloc[i + future]["close"])

            change = (
                future_close - current_close
            ) / current_close

            if change >= target_profit:

                target = 1

            elif change <= -target_profit:

                target = -1

            else:

                target = 0

            row["target"] = target

            row["future_return"] = change

                # ====================================================
    # BUILD DATASET
    # ====================================================

    def build(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Полное построение обучающего датасета.
        """

        self.build_features_dataset(df)

        self.generate_targets(df)

        dataset = pd.DataFrame(self.dataset)

        if self.config.drop_nan:

            dataset = dataset.dropna().reset_index(drop=True)

        return dataset

        # ====================================================
# PUBLIC API
# ====================================================

def build_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Быстрое построение обучающего датасета.
    """

    builder = DatasetBuilder()

    return builder.build(df)


__all__ = [
    "DatasetConfig",
    "DatasetBuilder",
    "build_dataset",
]