from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


VALID_SIGNALS = {"BUY", "SELL", "HOLD"}


@dataclass(frozen=True)
class SignalQualitySnapshot:
    total_signals: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    trade_signals: int
    buy_rate: float
    sell_rate: float
    hold_rate: float
    trade_rate: float
    average_confidence: float
    average_trade_confidence: float
    approved_trade_rate: float
    rejected_trade_rate: float
    ai_ml_agreement_rate: float
    ai_ml_conflict_rate: float
    ml_hold_rate: float
    invalid_signal_count: int
    diagnostic_flags: tuple[str, ...] = field(default_factory=tuple)


class SignalQualityAnalyzer:
    def __init__(
        self,
        *,
        min_confidence: float = 60.0,
        hold_warning_rate: float = 70.0,
        min_trade_rate: float = 5.0,
        max_trade_rate: float = 80.0,
        min_sample_size: int = 20,
    ) -> None:
        if not 0.0 <= min_confidence <= 100.0:
            raise ValueError("min_confidence must be between 0 and 100.")

        if not 0.0 <= hold_warning_rate <= 100.0:
            raise ValueError("hold_warning_rate must be between 0 and 100.")

        if not 0.0 <= min_trade_rate <= 100.0:
            raise ValueError("min_trade_rate must be between 0 and 100.")

        if not 0.0 <= max_trade_rate <= 100.0:
            raise ValueError("max_trade_rate must be between 0 and 100.")

        if min_trade_rate > max_trade_rate:
            raise ValueError("min_trade_rate cannot exceed max_trade_rate.")

        if min_sample_size < 1:
            raise ValueError("min_sample_size must be positive.")

        self.min_confidence = float(min_confidence)
        self.hold_warning_rate = float(hold_warning_rate)
        self.min_trade_rate = float(min_trade_rate)
        self.max_trade_rate = float(max_trade_rate)
        self.min_sample_size = int(min_sample_size)
        self._records: list[dict[str, Any]] = []

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default

        return number if number == number else default

    @staticmethod
    def _signal(value: Any) -> str:
        signal = str(value or "HOLD").strip().upper()

        aliases = {
            "LONG": "BUY",
            "SHORT": "SELL",
            "NEUTRAL": "HOLD",
            "WAIT": "HOLD",
        }

        return aliases.get(signal, signal)

    @staticmethod
    def _read(
        record: Any,
        name: str,
        default: Any = None,
    ) -> Any:
        if isinstance(record, Mapping):
            return record.get(name, default)

        return getattr(record, name, default)

    def add(
        self,
        signal: Any,
        *,
        confidence: float | None = None,
        ai_signal: Any = None,
        ai_confidence: float | None = None,
        ml_signal: Any = None,
        ml_probability: float | None = None,
        trade_approved: bool | None = None,
        timestamp: Any = None,
        timeframe: str | None = None,
        symbol: str | None = None,
        strategy: str | None = None,
    ) -> None:
        normalized_signal = self._signal(signal)

        self._records.append(
            {
                "signal": normalized_signal,
                "confidence": self._number(confidence),
                "ai_signal": (
                    self._signal(ai_signal)
                    if ai_signal is not None
                    else None
                ),
                "ai_confidence": (
                    self._number(ai_confidence)
                    if ai_confidence is not None
                    else None
                ),
                "ml_signal": (
                    self._signal(ml_signal)
                    if ml_signal is not None
                    else None
                ),
                "ml_probability": (
                    self._number(ml_probability)
                    if ml_probability is not None
                    else None
                ),
                "trade_approved": trade_approved,
                "timestamp": timestamp,
                "timeframe": timeframe,
                "symbol": symbol,
                "strategy": strategy,
            }
        )

    def add_result(
        self,
        result: Any,
        *,
        timestamp: Any = None,
        timeframe: str | None = None,
        symbol: str | None = None,
        strategy: str | None = None,
    ) -> None:
        self.add(
            self._read(result, "signal", "HOLD"),
            confidence=self._read(result, "confidence"),
            ai_signal=self._read(result, "ai_signal"),
            ai_confidence=self._read(result, "ai_confidence"),
            ml_signal=self._read(result, "ml_signal"),
            ml_probability=self._read(result, "ml_probability"),
            trade_approved=self._read(result, "trade_approved"),
            timestamp=(
                timestamp
                if timestamp is not None
                else self._read(result, "timestamp")
            ),
            timeframe=timeframe,
            symbol=symbol,
            strategy=strategy,
        )

    def add_many(self, records: Iterable[Any]) -> int:
        count = 0

        for record in records:
            if isinstance(record, Mapping) and "result" in record:
                self.add_result(
                    record["result"],
                    timestamp=record.get("timestamp"),
                    timeframe=record.get("timeframe"),
                    symbol=record.get("symbol"),
                    strategy=record.get("strategy"),
                )
            elif hasattr(record, "signal") and not isinstance(
                record,
                Mapping,
            ):
                self.add_result(record)
            else:
                self.add(
                    self._read(record, "signal", "HOLD"),
                    confidence=self._read(record, "confidence"),
                    ai_signal=self._read(record, "ai_signal"),
                    ai_confidence=self._read(
                        record,
                        "ai_confidence",
                    ),
                    ml_signal=self._read(record, "ml_signal"),
                    ml_probability=self._read(
                        record,
                        "ml_probability",
                    ),
                    trade_approved=self._read(
                        record,
                        "trade_approved",
                    ),
                    timestamp=self._read(record, "timestamp"),
                    timeframe=self._read(record, "timeframe"),
                    symbol=self._read(record, "symbol"),
                    strategy=self._read(record, "strategy"),
                )

            count += 1

        return count

    def clear(self) -> None:
        self._records.clear()

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(record)
            for record in self._records
        )

    def analyze(self) -> SignalQualitySnapshot:
        total = len(self._records)

        counts = Counter(
            record["signal"]
            for record in self._records
        )

        buy = counts["BUY"]
        sell = counts["SELL"]
        hold = counts["HOLD"]

        invalid = total - buy - sell - hold
        trades = buy + sell

        confidence_values = [
            record["confidence"]
            for record in self._records
        ]

        trade_confidences = [
            record["confidence"]
            for record in self._records
            if record["signal"] in {"BUY", "SELL"}
        ]

        approved_values = [
            record["trade_approved"]
            for record in self._records
            if record["trade_approved"] is not None
        ]

        agreement_values: list[bool] = []
        conflict_values: list[bool] = []
        ml_hold_values: list[bool] = []

        for record in self._records:
            ai = record["ai_signal"]
            ml = record["ml_signal"]

            if ai in VALID_SIGNALS and ml in VALID_SIGNALS:
                agreement_values.append(ai == ml)
                conflict_values.append(ai != ml)

            if ml is not None:
                ml_hold_values.append(ml == "HOLD")

        hold_rate = self._rate(
            hold,
            total,
        )

        trade_rate = self._rate(
            trades,
            total,
        )

        flags: list[str] = []

        if total >= self.min_sample_size:
            if hold_rate >= self.hold_warning_rate:
                flags.append("HIGH_HOLD_RATE")

            if trade_rate < self.min_trade_rate:
                flags.append("UNDERTRADING")
            elif trade_rate > self.max_trade_rate:
                flags.append("OVERTRADING")

        if invalid:
            flags.append("INVALID_SIGNALS")

        if approved_values:
            approved_rate = self._rate(
                sum(bool(value) for value in approved_values),
                len(approved_values),
            )
        else:
            approved_rate = 0.0

        if (
            total >= self.min_sample_size
            and not approved_values
            and trades
        ):
            flags.append("MISSING_TRADE_APPROVAL_DATA")

        return SignalQualitySnapshot(
            total_signals=total,
            buy_signals=buy,
            sell_signals=sell,
            hold_signals=hold,
            trade_signals=trades,
            buy_rate=self._rate(buy, total),
            sell_rate=self._rate(sell, total),
            hold_rate=hold_rate,
            trade_rate=trade_rate,
            average_confidence=self._average(
                confidence_values,
            ),
            average_trade_confidence=self._average(
                trade_confidences,
            ),
            approved_trade_rate=approved_rate,
            rejected_trade_rate=(
                100.0 - approved_rate
                if approved_values
                else 0.0
            ),
            ai_ml_agreement_rate=self._rate(
                sum(agreement_values),
                len(agreement_values),
            ),
            ai_ml_conflict_rate=self._rate(
                sum(conflict_values),
                len(conflict_values),
            ),
            ml_hold_rate=self._rate(
                sum(ml_hold_values),
                len(ml_hold_values),
            ),
            invalid_signal_count=invalid,
            diagnostic_flags=tuple(flags),
        )

    def diagnose(self) -> list[str]:
        snapshot = self.analyze()
        diagnoses: list[str] = []

        if "HIGH_HOLD_RATE" in snapshot.diagnostic_flags:
            diagnoses.append(
                "HOLD dominates the signal distribution."
            )

        if "UNDERTRADING" in snapshot.diagnostic_flags:
            diagnoses.append(
                "Trade signal frequency is below the configured range."
            )

        if "OVERTRADING" in snapshot.diagnostic_flags:
            diagnoses.append(
                "Trade signal frequency is above the configured range."
            )

        if "INVALID_SIGNALS" in snapshot.diagnostic_flags:
            diagnoses.append(
                "One or more records contain unsupported signal values."
            )

        if "MISSING_TRADE_APPROVAL_DATA" in snapshot.diagnostic_flags:
            diagnoses.append(
                "Trade approval data is missing for a meaningful sample."
            )

        if snapshot.ai_ml_conflict_rate > 50.0:
            diagnoses.append(
                "AI and ML signals conflict more often than they agree."
            )

        if (
            snapshot.ml_hold_rate >= self.hold_warning_rate
            and snapshot.total_signals >= self.min_sample_size
        ):
            diagnoses.append(
                "ML HOLD probability is a major contributor "
                "to limited trade activity."
            )

        return diagnoses

    def grouped_summary(
        self,
        field: str,
    ) -> dict[str, SignalQualitySnapshot]:
        if field not in {
            "timeframe",
            "symbol",
            "strategy",
        }:
            raise ValueError(
                "field must be timeframe, symbol, or strategy."
            )

        groups: dict[str, list[dict[str, Any]]] = {}

        for record in self._records:
            key = str(
                record.get(field)
                or "UNKNOWN"
            )

            groups.setdefault(
                key,
                [],
            ).append(record)

        summaries: dict[str, SignalQualitySnapshot] = {}

        for key, records in groups.items():
            analyzer = SignalQualityAnalyzer(
                min_confidence=self.min_confidence,
                hold_warning_rate=self.hold_warning_rate,
                min_trade_rate=self.min_trade_rate,
                max_trade_rate=self.max_trade_rate,
                min_sample_size=self.min_sample_size,
            )

            analyzer._records = [
                dict(record)
                for record in records
            ]

            summaries[key] = analyzer.analyze()

        return summaries

    @staticmethod
    def _rate(
        value: int,
        total: int,
    ) -> float:
        if total <= 0:
            return 0.0

        return round(
            value / total * 100.0,
            4,
        )

    @staticmethod
    def _average(
        values: Sequence[float],
    ) -> float:
        if not values:
            return 0.0

        return round(
            sum(values) / len(values),
            4,
        )


def analyze_signal_quality(
    records: Iterable[Any],
    **kwargs: Any,
) -> SignalQualitySnapshot:
    analyzer = SignalQualityAnalyzer(**kwargs)
    analyzer.add_many(records)
    return analyzer.analyze()


__all__ = [
    "SignalQualityAnalyzer",
    "SignalQualitySnapshot",
    "analyze_signal_quality",
]