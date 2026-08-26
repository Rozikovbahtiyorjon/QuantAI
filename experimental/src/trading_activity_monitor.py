from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class TradingActivityRecord:
    signal: str
    timestamp: Any = None
    executed: bool = False
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    confidence: float = 0.0
    quality: Optional[float] = None
    pnl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_signal(self) -> str:
        value = str(self.signal).strip().upper()
        if value not in {"BUY", "SELL", "HOLD"}:
            raise ValueError(f"Unsupported signal: {self.signal!r}")
        return value


@dataclass(frozen=True)
class TradingActivitySnapshot:
    total_signals: int
    executed_trades: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    execution_rate: float
    trade_rate: float
    average_confidence: float
    average_quality: Optional[float]
    total_pnl: Optional[float]
    min_trades: int
    max_trades: int
    activity_status: str
    diagnostics: List[str]


class TradingActivityMonitor:
    def __init__(
        self,
        min_trades: int = 1,
        max_trades: int = 20,
        min_confidence: float = 0.0,
        min_quality: float = 0.0,
    ) -> None:
        if min_trades < 0:
            raise ValueError("min_trades must be non-negative")
        if max_trades < min_trades:
            raise ValueError("max_trades must be >= min_trades")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if not 0.0 <= min_quality <= 1.0:
            raise ValueError("min_quality must be between 0 and 1")

        self.min_trades = int(min_trades)
        self.max_trades = int(max_trades)
        self.min_confidence = float(min_confidence)
        self.min_quality = float(min_quality)
        self._records: List[TradingActivityRecord] = []

    @property
    def records(self) -> List[TradingActivityRecord]:
        return list(self._records)

    def reset(self) -> None:
        self._records.clear()

    def add_record(
        self,
        record: Optional[TradingActivityRecord] = None,
        **kwargs: Any,
    ) -> TradingActivityRecord:
        if record is not None and kwargs:
            raise ValueError("Provide either record or keyword fields, not both")

        if record is None:
            record = TradingActivityRecord(**kwargs)

        record.normalized_signal()

        if not 0.0 <= float(record.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if record.quality is not None and not 0.0 <= float(record.quality) <= 1.0:
            raise ValueError("quality must be between 0 and 1")

        self._records.append(record)
        return record

    def extend(self, records: Iterable[TradingActivityRecord]) -> None:
        for record in records:
            self.add_record(record)

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _average(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _status(self, executed_trades: int) -> str:
        if executed_trades < self.min_trades:
            return "UNDERACTIVE"

        if executed_trades > self.max_trades:
            return "OVERACTIVE"

        return "NORMAL"

    def diagnostics(self) -> List[str]:
        records = self._records
        diagnostics: List[str] = []

        executed = sum(bool(record.executed) for record in records)

        if executed < self.min_trades:
            diagnostics.append("TRADE_COUNT_BELOW_MINIMUM")

        if executed > self.max_trades:
            diagnostics.append("TRADE_COUNT_ABOVE_MAXIMUM")

        if records:
            low_confidence = sum(
                float(record.confidence) < self.min_confidence
                for record in records
            )

            if low_confidence:
                diagnostics.append("LOW_CONFIDENCE_SIGNALS")

            quality_values = [
                float(record.quality)
                for record in records
                if record.quality is not None
            ]

            if quality_values:
                low_quality = sum(
                    value < self.min_quality
                    for value in quality_values
                )

                if low_quality:
                    diagnostics.append("LOW_SIGNAL_QUALITY")

        if not records:
            diagnostics.append("NO_SIGNALS")

        return diagnostics

    def snapshot(self) -> TradingActivitySnapshot:
        records = self._records

        total = len(records)
        executed = sum(bool(record.executed) for record in records)

        buys = sum(
            record.normalized_signal() == "BUY"
            for record in records
        )

        sells = sum(
            record.normalized_signal() == "SELL"
            for record in records
        )

        holds = sum(
            record.normalized_signal() == "HOLD"
            for record in records
        )

        confidences = [
            float(record.confidence)
            for record in records
        ]

        qualities = [
            float(record.quality)
            for record in records
            if record.quality is not None
        ]

        pnl_values = [
            float(record.pnl)
            for record in records
            if record.pnl is not None
        ]

        average_quality = (
            self._average(qualities)
            if qualities
            else None
        )

        total_pnl = (
            sum(pnl_values)
            if pnl_values
            else None
        )

        rate = self._rate(executed, total)

        return TradingActivitySnapshot(
            total_signals=total,
            executed_trades=executed,
            buy_signals=buys,
            sell_signals=sells,
            hold_signals=holds,
            execution_rate=rate,
            trade_rate=rate,
            average_confidence=self._average(confidences),
            average_quality=average_quality,
            total_pnl=total_pnl,
            min_trades=self.min_trades,
            max_trades=self.max_trades,
            activity_status=self._status(executed),
            diagnostics=self.diagnostics(),
        )

    def analyze(self) -> Dict[str, Any]:
        snapshot = self.snapshot()

        return {
            "total_signals": snapshot.total_signals,
            "executed_trades": snapshot.executed_trades,
            "buy_signals": snapshot.buy_signals,
            "sell_signals": snapshot.sell_signals,
            "hold_signals": snapshot.hold_signals,
            "execution_rate": snapshot.execution_rate,
            "trade_rate": snapshot.trade_rate,
            "average_confidence": snapshot.average_confidence,
            "average_quality": snapshot.average_quality,
            "total_pnl": snapshot.total_pnl,
            "activity_status": snapshot.activity_status,
            "diagnostics": list(snapshot.diagnostics),
            "trade_range": {
                "min": snapshot.min_trades,
                "max": snapshot.max_trades,
            },
        }

    def analyze_window(
        self,
        start: Any = None,
        end: Any = None,
    ) -> TradingActivitySnapshot:
        selected = [
            record
            for record in self._records
            if self._in_range(record.timestamp, start, end)
        ]

        monitor = TradingActivityMonitor(
            min_trades=self.min_trades,
            max_trades=self.max_trades,
            min_confidence=self.min_confidence,
            min_quality=self.min_quality,
        )

        monitor.extend(selected)

        return monitor.snapshot()

    @staticmethod
    def _in_range(
        timestamp: Any,
        start: Any,
        end: Any,
    ) -> bool:
        if timestamp is None:
            return start is None and end is None

        if start is not None and timestamp < start:
            return False

        if end is not None and timestamp > end:
            return False

        return True

    def daily_snapshots(self) -> Dict[Any, TradingActivitySnapshot]:
        groups: Dict[Any, List[TradingActivityRecord]] = {}

        for record in self._records:
            timestamp = record.timestamp

            if isinstance(timestamp, datetime):
                key = timestamp.date()
            elif isinstance(timestamp, date):
                key = timestamp
            else:
                key = str(timestamp)[:10]

            groups.setdefault(key, []).append(record)

        result: Dict[Any, TradingActivitySnapshot] = {}

        for key, records in groups.items():
            monitor = TradingActivityMonitor(
                min_trades=self.min_trades,
                max_trades=self.max_trades,
                min_confidence=self.min_confidence,
                min_quality=self.min_quality,
            )

            monitor.extend(records)
            result[key] = monitor.snapshot()

        return result