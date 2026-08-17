from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ChampionGovernanceSnapshot:
    status: str
    candidate_id: str
    champion_id: str | None
    evaluated: bool
    qualified: bool
    admission_action: str
    transition_action: str
    transition_changed: bool
    stable: bool
    replacement_allowed: bool
    rollback_action: str
    rollback_allowed: bool
    candidate_score: float
    champion_score: float
    improvement: float
    reason: str


class ChampionGovernanceEngine:
    """Single governance boundary for controlled Champion promotion."""

    EVALUATOR_METRICS = {
        "profit_factor",
        "net_profit",
        "win_rate",
        "sharpe_ratio",
        "max_drawdown",
    }

    FEEDBACK_METRICS = {
        "net_profit",
        "win_rate",
        "trade_count",
        "max_drawdown",
        "signal_quality",
        "stability",
    }

    def __init__(
        self,
        evaluator: Any,
        admission_controller: Any,
        transition_decision: Any,
        transition_executor: Any,
        stability_monitor: Any,
        replacement_guard: Any,
        rollback_guard: Any,
        performance_feedback: Any | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.admission_controller = admission_controller
        self.transition_decision = transition_decision
        self.transition_executor = transition_executor
        self.stability_monitor = stability_monitor
        self.replacement_guard = replacement_guard
        self.rollback_guard = rollback_guard
        self.performance_feedback = performance_feedback

    @staticmethod
    def _read(
        value: Any,
        *names: str,
        default: Any = None,
    ) -> Any:
        if value is None:
            return default

        if isinstance(value, Mapping):
            for name in names:
                if name in value:
                    return value[name]

            return default

        for name in names:
            if hasattr(value, name):
                return getattr(value, name)

        return default

    @staticmethod
    def _bool(
        value: Any,
        *names: str,
        default: bool = False,
    ) -> bool:
        return bool(
            ChampionGovernanceEngine._read(
                value,
                *names,
                default=default,
            )
        )

    @staticmethod
    def _float(
        value: Any,
        *names: str,
        default: float = 0.0,
    ) -> float:
        result = ChampionGovernanceEngine._read(
            value,
            *names,
            default=default,
        )

        return float(result)

    @staticmethod
    def _validate_metrics(
        metrics: Mapping[str, Any],
        name: str,
    ) -> dict[str, Any]:
        if metrics is None:
            raise ValueError(
                f"{name} metrics must not be None."
            )

        if not isinstance(metrics, Mapping):
            raise TypeError(
                f"{name} metrics must be a mapping."
            )

        return dict(metrics)

    @classmethod
    def _evaluator_metrics(
        cls,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        missing = cls.EVALUATOR_METRICS - set(metrics)

        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(
                "Missing evaluator metrics: "
                + missing_names
            )

        return {
            name: metrics[name]
            for name in cls.EVALUATOR_METRICS
        }

    @classmethod
    def _feedback_metrics(
        cls,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not cls.FEEDBACK_METRICS.issubset(metrics):
            return None

        return {
            name: metrics[name]
            for name in cls.FEEDBACK_METRICS
        }

    def _record_feedback(
        self,
        candidate_id: str,
        candidate_metrics: Mapping[str, Any],
        champion_id: str | None,
        champion_metrics: Mapping[str, Any],
    ) -> None:
        if self.performance_feedback is None:
            return

        record = getattr(
            self.performance_feedback,
            "record",
            None,
        )

        if not callable(record):
            return

        candidate_feedback = self._feedback_metrics(
            candidate_metrics
        )

        if candidate_feedback is not None:
            record(
                candidate_id,
                candidate_feedback,
            )

        if champion_id and champion_metrics:
            champion_feedback = self._feedback_metrics(
                champion_metrics
            )

            if champion_feedback is not None:
                record(
                    champion_id,
                    champion_feedback,
                )

    def run(
        self,
        candidate_id: str,
        champion_id: str | None,
        candidate_metrics: Mapping[str, Any],
        champion_metrics: Mapping[str, Any] | None = None,
        samples: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> ChampionGovernanceSnapshot:
        if not candidate_id:
            raise ValueError(
                "candidate_id must not be empty."
            )

        if samples is not None and samples < 0:
            raise ValueError(
                "samples must be non-negative."
            )

        candidate = self._validate_metrics(
            candidate_metrics,
            "candidate",
        )

        champion = self._validate_metrics(
            champion_metrics or {},
            "champion",
        )

        context = dict(context or {})

        self._record_feedback(
            candidate_id,
            candidate,
            champion_id,
            champion,
        )

        evaluated = bool(champion)
        qualified = True

        candidate_score = 0.0
        champion_score = 0.0
        improvement = 0.0

        if evaluated:
            evaluation = self.evaluator.evaluate(
                self._evaluator_metrics(candidate),
                self._evaluator_metrics(champion),
            )

            qualified = self._bool(
                evaluation,
                "qualified",
                default=False,
            )

            candidate_score = self._float(
                evaluation,
                "candidate_score",
                default=0.0,
            )

            champion_score = self._float(
                evaluation,
                "champion_score",
                default=0.0,
            )

            improvement = self._float(
                evaluation,
                "improvement",
                default=0.0,
            )

            if not qualified:
                return ChampionGovernanceSnapshot(
                    status="REJECTED",
                    candidate_id=candidate_id,
                    champion_id=champion_id,
                    evaluated=True,
                    qualified=False,
                    admission_action="REJECT",
                    transition_action="REJECT",
                    transition_changed=False,
                    stable=False,
                    replacement_allowed=False,
                    rollback_action="HOLD",
                    rollback_allowed=False,
                    candidate_score=candidate_score,
                    champion_score=champion_score,
                    improvement=improvement,
                    reason=(
                        "candidate evaluation "
                        "did not qualify"
                    ),
                )

        admission = (
            self.admission_controller.evaluate(
                candidate,
                champion,
                samples=samples,
            )
        )

        admission_action = str(
            self._read(
                admission,
                "action",
                "decision",
                default="REJECT",
            )
        ).upper()

        if admission_action != "ADMIT":
            transition_action = (
                "HOLD"
                if admission_action == "HOLD"
                else "REJECT"
            )

            return ChampionGovernanceSnapshot(
                status=admission_action,
                candidate_id=candidate_id,
                champion_id=champion_id,
                evaluated=evaluated,
                qualified=qualified,
                admission_action=admission_action,
                transition_action=transition_action,
                transition_changed=False,
                stable=False,
                replacement_allowed=False,
                rollback_action="HOLD",
                rollback_allowed=False,
                candidate_score=candidate_score,
                champion_score=champion_score,
                improvement=improvement,
                reason=str(
                    self._read(
                        admission,
                        "reason",
                        default="admission denied",
                    )
                ),
            )

        transition = self.transition_decision.decide(
            champion,
            candidate,
        )

        transition_action = str(
            self._read(
                transition,
                "action",
                "decision",
                default="REJECT",
            )
        ).upper()

        if transition_action not in {
            "PROMOTE",
            "REPLACE",
        }:
            return ChampionGovernanceSnapshot(
                status=transition_action,
                candidate_id=candidate_id,
                champion_id=champion_id,
                evaluated=evaluated,
                qualified=qualified,
                admission_action=admission_action,
                transition_action=transition_action,
                transition_changed=False,
                stable=self._bool(
                    transition,
                    "stable",
                    default=False,
                ),
                replacement_allowed=False,
                rollback_action="HOLD",
                rollback_allowed=False,
                candidate_score=candidate_score,
                champion_score=champion_score,
                improvement=improvement,
                reason=str(
                    self._read(
                        transition,
                        "reason",
                        default="transition not approved",
                    )
                ),
            )

        stability = self.stability_monitor.analyze(
            candidate,
            champion,
        )

        stable = (
            self._read(
                stability,
                "status",
                default="WARNING",
            )
            == "STABLE"
        )

        replacement = None

        if champion:
            replacement = (
                self.replacement_guard.evaluate(
                    champion,
                    candidate,
                )
            )

        replacement_allowed = (
            self._bool(
                replacement,
                "approved",
                "allowed",
                default=True,
            )
            if replacement is not None
            else True
        )

        if not stable or not replacement_allowed:
            if not stable:
                status = "DEGRADED"
                reason = (
                    "candidate stability "
                    "below threshold"
                )
            else:
                status = "REJECTED"
                reason = (
                    "replacement guard "
                    "rejected candidate"
                )

            return ChampionGovernanceSnapshot(
                status=status,
                candidate_id=candidate_id,
                champion_id=champion_id,
                evaluated=evaluated,
                qualified=qualified,
                admission_action=admission_action,
                transition_action=transition_action,
                transition_changed=False,
                stable=stable,
                replacement_allowed=replacement_allowed,
                rollback_action="HOLD",
                rollback_allowed=False,
                candidate_score=candidate_score,
                champion_score=champion_score,
                improvement=improvement,
                reason=reason,
            )

        transition_result = (
            self.transition_executor.execute(
                transition,
                champion,
                candidate,
            )
        )

        transition_changed = self._bool(
            transition_result,
            "changed",
            "transitioned",
            default=False,
        )

        rollback = (
            self.rollback_guard.evaluate(
                candidate,
                champion,
                samples=samples,
            )
        )

        rollback_action = str(
            self._read(
                rollback,
                "action",
                "decision",
                default="HOLD",
            )
        ).upper()

        rollback_allowed = (
            rollback_action == "ROLLBACK"
        )

        if rollback_allowed:
            status = "ROLLBACK"
            reason = str(
                self._read(
                    rollback,
                    "reason",
                    default=(
                        "performance degradation "
                        "detected"
                    ),
                )
            )
        elif not transition_changed:
            status = "REJECTED"
            reason = str(
                self._read(
                    transition_result,
                    "reason",
                    default=(
                        "transition executor "
                        "did not change champion"
                    ),
                )
            )
        else:
            status = "PROMOTED"
            reason = str(
                self._read(
                    transition_result,
                    "reason",
                    default="candidate promoted",
                )
            )

        return ChampionGovernanceSnapshot(
            status=status,
            candidate_id=candidate_id,
            champion_id=champion_id,
            evaluated=evaluated,
            qualified=qualified,
            admission_action=admission_action,
            transition_action=transition_action,
            transition_changed=transition_changed,
            stable=stable,
            replacement_allowed=replacement_allowed,
            rollback_action=rollback_action,
            rollback_allowed=rollback_allowed,
            candidate_score=candidate_score,
            champion_score=champion_score,
            improvement=improvement,
            reason=reason,
        )

    def evaluate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ChampionGovernanceSnapshot:
        return self.run(*args, **kwargs)


__all__ = [
    "ChampionGovernanceEngine",
    "ChampionGovernanceSnapshot",
]