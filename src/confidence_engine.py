"""
====================================================
QuantAI Professional AI Trading System
Confidence Engine v3.0
====================================================

Назначение:
Оценка качества торгового сигнала на основе
нескольких независимых аналитических модулей.

Каждый модуль возвращает собственную оценку.

Confidence Engine объединяет их
в итоговую вероятность сделки.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ====================================================
# Score Component
# ====================================================

@dataclass
class ScoreComponent:
    """
    Оценка одного аналитического модуля.
    """

    name: str

    score: float

    weight: float = 1.0

    description: str = ""


# ====================================================
# Confidence Result
# ====================================================

@dataclass
class ConfidenceResult:
    """
    Финальный результат Confidence Engine.
    """

    total_score: float

    confidence: float

    probability: float

    decision: str

    components: List[ScoreComponent] = field(default_factory=list)

    reasons: List[str] = field(default_factory=list)

    # ====================================================
# Confidence Engine
# ====================================================

class ConfidenceEngine:
    """
    Центральный AI Engine проекта QuantAI.

    Собирает оценки всех аналитических модулей.

    Затем вычисляет:

        Total Score

        Confidence

        Probability

        Decision
    """

    def __init__(self):

        self.components: List[ScoreComponent] = []

        self.weights: Dict[str, float] = {

            "trend": 1.50,

            "momentum": 1.20,

            "volume": 1.10,

            "volatility": 1.00,

            "liquidity": 1.40,

            "structure": 1.30,

            "regime": 1.50,

        }

    # =================================================

    def reset(self):

        self.components.clear()

    # =================================================

    def add_component(

        self,

        name: str,

        score: float,

        description: str = "",

    ):

        weight = self.weights.get(

            name.lower(),

            1.0,

        )

        self.components.append(

            ScoreComponent(

                name=name,

                score=float(score),

                weight=weight,

                description=description,

            )

        )    # =================================================

    def calculate_score(self) -> float:
        """
        Рассчитывает общий взвешенный Score.
        """

        if len(self.components) == 0:
            return 0.0

        weighted_sum = 0.0

        total_weight = 0.0

        for component in self.components:

            weighted_sum += (
                component.score *
                component.weight
            )

            total_weight += component.weight

        if total_weight == 0:
            return 0.0

        return round(
            weighted_sum / total_weight,
            2,
        )

    # =================================================

    def calculate_confidence(
        self,
        score: float,
    ) -> float:
        """
        Преобразует Score
        в Confidence (%).
        """

        confidence = 50 + score * 10

        confidence = max(0, confidence)

        confidence = min(100, confidence)

        return round(confidence, 2)

    # =================================================

    def calculate_probability(
        self,
        confidence: float,
    ) -> float:
        """
        Вероятность сделки.
        """

        return round(confidence, 2)

            # =================================================

    def decide(
        self,
        score: float,
        confidence: float,
    ) -> str:
        """
        Финальное решение AI.
        """

        if confidence < 60:
            return "HOLD"

        if score >= 1.0:
            return "BUY"

        if score <= -1.0:
            return "SELL"

        return "HOLD"

    # =================================================

    def evaluate(self) -> ConfidenceResult:
        """
        Полный расчёт Confidence Engine.
        """

        score = self.calculate_score()

        confidence = self.calculate_confidence(
            score,
        )

        probability = self.calculate_probability(
            confidence,
        )

        decision = self.decide(
            score,
            confidence,
        )

        reasons = []

        for component in self.components:

            reasons.append(

                f"{component.name}: "

                f"{component.score:.2f}"

            )

        return ConfidenceResult(

            total_score=score,

            confidence=confidence,

            probability=probability,

            decision=decision,

            components=self.components.copy(),

            reasons=reasons,

        )

            # =================================================

    def summary(self) -> str:
        """
        Краткое текстовое представление результата.
        """

        result = self.evaluate()

        return (
            f"Decision={result.decision} | "
            f"Score={result.total_score:.2f} | "
            f"Confidence={result.confidence:.2f}%"
        )

    # =================================================

    def print_report(self):
        """
        Печать полного отчёта Confidence Engine.
        """

        result = self.evaluate()

        print()

        print("=" * 60)
        print("CONFIDENCE ENGINE")
        print("=" * 60)

        print(f"Decision      : {result.decision}")
        print(f"Score         : {result.total_score:.2f}")
        print(f"Confidence    : {result.confidence:.2f}%")
        print(f"Probability   : {result.probability:.2f}%")

        print()

        print("Components:")

        for component in result.components:

            print(
                f"{component.name:<15}"
                f"{component.score:>7.2f}"
                f"   w={component.weight:.2f}"
            )

        print("=" * 60)

        # ====================================================
# Module Export
# ====================================================

__all__ = [
    "ScoreComponent",
    "ConfidenceResult",
    "ConfidenceEngine",
]