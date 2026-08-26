from dataclasses import dataclass
from typing import Iterable, Mapping

import math
import re


@dataclass(frozen=True)
class SentimentResult:
    score: float
    confidence: float
    signal: str
    positive_count: int
    negative_count: int
    neutral_count: int
    source_count: int


class SentimentAnalysisEngine:
    """
    Deterministic crypto-market sentiment engine.
    """

    DEFAULT_POSITIVE_TERMS = {
        "bullish": 1.0,
        "bull": 0.9,
        "breakout": 0.9,
        "breaks": 0.7,
        "break": 0.7,
        "rally": 0.9,
        "surge": 1.0,
        "surges": 1.0,
        "soar": 1.0,
        "soars": 1.0,
        "pump": 0.8,
        "pumping": 0.9,
        "growth": 0.7,
        "gains": 0.7,
        "gain": 0.7,
        "strong": 0.6,
        "strength": 0.6,
        "positive": 0.6,
        "optimistic": 0.8,
        "adoption": 0.8,
        "adopt": 0.7,
        "inflow": 0.7,
        "inflows": 0.7,
        "approval": 0.9,
        "approved": 0.9,
        "partnership": 0.7,
        "partnerships": 0.7,
        "upgrade": 0.7,
        "upgraded": 0.7,
        "record": 0.7,
        "high": 0.5,
        "ath": 0.9,
        "accumulate": 0.8,
        "accumulation": 0.8,
    }

    DEFAULT_NEGATIVE_TERMS = {
        "bearish": 1.0,
        "bear": 0.9,
        "breakdown": 0.9,
        "crash": 1.0,
        "crashes": 1.0,
        "dump": 0.9,
        "dumping": 1.0,
        "selloff": 1.0,
        "sell-off": 1.0,
        "decline": 0.7,
        "declines": 0.7,
        "drop": 0.7,
        "drops": 0.7,
        "fall": 0.7,
        "falls": 0.7,
        "weak": 0.6,
        "weakness": 0.6,
        "negative": 0.6,
        "pessimistic": 0.8,
        "outflow": 0.7,
        "outflows": 0.7,
        "rejection": 0.8,
        "rejected": 0.8,
        "hack": 1.0,
        "hacked": 1.0,
        "exploit": 1.0,
        "exploited": 1.0,
        "liquidation": 0.9,
        "liquidations": 0.9,
        "regulation": 0.4,
        "ban": 1.0,
        "banned": 1.0,
        "lawsuit": 0.8,
        "loss": 0.7,
        "losses": 0.7,
        "low": 0.5,
    }

    NEGATION_TERMS = {
        "not",
        "no",
        "never",
        "without",
        "unlikely",
        "isnt",
        "isn't",
        "wasnt",
        "wasn't",
        "dont",
        "don't",
        "doesnt",
        "doesn't",
    }

    INTENSIFIERS = {
        "very": 1.25,
        "extremely": 1.50,
        "strongly": 1.35,
        "highly": 1.35,
        "massive": 1.50,
        "major": 1.25,
        "significant": 1.25,
    }

    def __init__(
        self,
        positive_terms: Mapping[str, float] | None = None,
        negative_terms: Mapping[str, float] | None = None,
        bullish_threshold: float = 0.15,
        bearish_threshold: float = -0.15,
        min_confidence: float = 0.20,
        negation_window: int = 3,
    ) -> None:
        self.positive_terms = (
            dict(self.DEFAULT_POSITIVE_TERMS)
            if positive_terms is None
            else dict(positive_terms)
        )
        self.negative_terms = (
            dict(self.DEFAULT_NEGATIVE_TERMS)
            if negative_terms is None
            else dict(negative_terms)
        )
        self.bullish_threshold = float(bullish_threshold)
        self.bearish_threshold = float(bearish_threshold)
        self.min_confidence = float(min_confidence)
        self.negation_window = int(negation_window)

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if self.bearish_threshold >= self.bullish_threshold:
            raise ValueError(
                "bearish_threshold must be lower than bullish_threshold."
            )

        if self.min_confidence < 0 or self.min_confidence > 1:
            raise ValueError("min_confidence must be between 0 and 1.")

        if self.negation_window < 0:
            raise ValueError("negation_window must be >= 0.")

        for terms in (
            self.positive_terms,
            self.negative_terms,
        ):
            if not terms:
                raise ValueError(
                    "Sentiment term dictionaries must not be empty."
                )

            for term, weight in terms.items():
                if not isinstance(term, str) or not term.strip():
                    raise ValueError(
                        "Sentiment terms must be non-empty strings."
                    )

                if not isinstance(weight, (int, float)):
                    raise TypeError(
                        "Sentiment weights must be numeric."
                    )

                if not math.isfinite(float(weight)):
                    raise ValueError(
                        "Sentiment weights must be finite."
                    )

                if float(weight) < 0:
                    raise ValueError(
                        "Sentiment weights must be non-negative."
                    )

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = str(text).lower()
        text = text.replace("—", " ")
        text = text.replace("–", " ")
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"[^a-z0-9' -]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not text:
            return []

        return text.split()

    def _term_weight(
        self,
        tokens: list[str],
        index: int,
        term: str,
        weight: float,
    ) -> float:
        term_tokens = self._tokenize(
            self._normalize_text(term)
        )

        if not term_tokens:
            return 0.0

        length = len(term_tokens)

        if tokens[index:index + length] != term_tokens:
            return 0.0

        start = max(0, index - self.negation_window)
        context = tokens[start:index]

        negated = any(
            token in self.NEGATION_TERMS
            for token in context
        )

        multiplier = 1.0

        for token in context[-2:]:
            if token in self.INTENSIFIERS:
                multiplier *= self.INTENSIFIERS[token]

        value = float(weight) * multiplier

        if negated:
            value *= -1.0

        return value

    def analyze_text(self, text: str) -> SentimentResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        normalized = self._normalize_text(text)
        tokens = self._tokenize(normalized)

        if not tokens:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                signal="NEUTRAL",
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                source_count=1,
            )

        positive_count = 0
        negative_count = 0

        positive_score = 0.0
        negative_score = 0.0

        matched_positions: set[int] = set()

        all_terms = [
            ("positive", term, weight)
            for term, weight in self.positive_terms.items()
        ] + [
            ("negative", term, weight)
            for term, weight in self.negative_terms.items()
        ]

        all_terms.sort(
            key=lambda item: len(
                self._tokenize(
                    self._normalize_text(item[1])
                )
            ),
            reverse=True,
        )

        for category, term, weight in all_terms:
            term_tokens = self._tokenize(
                self._normalize_text(term)
            )

            term_length = len(term_tokens)

            if not term_length:
                continue

            for index in range(
                len(tokens) - term_length + 1
            ):
                positions = set(
                    range(index, index + term_length)
                )

                if positions & matched_positions:
                    continue

                contribution = self._term_weight(
                    tokens=tokens,
                    index=index,
                    term=term,
                    weight=weight,
                )

                if contribution == 0.0:
                    continue

                matched_positions.update(positions)

                if category == "positive":
                    if contribution > 0:
                        positive_count += 1
                        positive_score += contribution
                    else:
                        negative_count += 1
                        negative_score += abs(contribution)
                else:
                    if contribution > 0:
                        negative_count += 1
                        negative_score += contribution
                    else:
                        positive_count += 1
                        positive_score += abs(contribution)

        total_signal = positive_score + negative_score

        if total_signal <= 1e-12:
            score = 0.0
            confidence = 0.0
        else:
            score = (
                (positive_score - negative_score)
                / total_signal
            )

            coverage = min(
                1.0,
                total_signal
                / max(2.0, len(tokens) * 0.25),
            )

            directional_strength = abs(score)

            confidence = min(
                1.0,
                directional_strength * 0.75
                + coverage * 0.25,
            )

        if confidence < self.min_confidence:
            signal = "NEUTRAL"
        elif score >= self.bullish_threshold:
            signal = "BULLISH"
        elif score <= self.bearish_threshold:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        matched_count = positive_count + negative_count
        neutral_count = max(
            0,
            len(tokens) - matched_count,
        )

        return SentimentResult(
            score=float(max(-1.0, min(1.0, score))),
            confidence=float(
                max(0.0, min(1.0, confidence))
            ),
            signal=signal,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            source_count=1,
        )

    def analyze(
        self,
        texts: Iterable[str],
    ) -> SentimentResult:
        if isinstance(texts, str):
            raise TypeError(
                "texts must be an iterable of strings, not a single string."
            )

        text_list = list(texts)

        if not text_list:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                signal="NEUTRAL",
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                source_count=0,
            )

        if any(
            not isinstance(text, str)
            for text in text_list
        ):
            raise TypeError(
                "All sentiment sources must be strings."
            )

        results = [
            self.analyze_text(text)
            for text in text_list
        ]

        total_weight = sum(
            max(result.confidence, 0.05)
            for result in results
        )

        weighted_score = sum(
            result.score
            * max(result.confidence, 0.05)
            for result in results
        ) / total_weight

        average_confidence = sum(
            result.confidence
            for result in results
        ) / len(results)

        positive_count = sum(
            result.positive_count
            for result in results
        )

        negative_count = sum(
            result.negative_count
            for result in results
        )

        neutral_count = sum(
            result.neutral_count
            for result in results
        )

        if average_confidence < self.min_confidence:
            signal = "NEUTRAL"
        elif weighted_score >= self.bullish_threshold:
            signal = "BULLISH"
        elif weighted_score <= self.bearish_threshold:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return SentimentResult(
            score=float(
                max(-1.0, min(1.0, weighted_score))
            ),
            confidence=float(
                max(
                    0.0,
                    min(1.0, average_confidence),
                )
            ),
            signal=signal,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            source_count=len(text_list),
        )

    def signal(self, texts: Iterable[str]) -> str:
        return self.analyze(texts).signal

    def compare(
        self,
        texts: Iterable[str],
        expected_signal: str,
    ) -> bool:
        if not isinstance(expected_signal, str):
            raise TypeError(
                "expected_signal must be a string."
            )

        return self.signal(texts) == expected_signal.upper()

    @staticmethod
    def summarize(
        result: SentimentResult,
    ) -> dict[str, float | int | str]:
        if not isinstance(result, SentimentResult):
            raise TypeError(
                "result must be a SentimentResult instance."
            )

        return {
            "score": float(result.score),
            "confidence": float(result.confidence),
            "signal": result.signal,
            "positive_count": int(result.positive_count),
            "negative_count": int(result.negative_count),
            "neutral_count": int(result.neutral_count),
            "source_count": int(result.source_count),
        }