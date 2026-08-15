from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Mapping


@dataclass(frozen=True)
class ChampionImprovementSnapshot:
    status: str
    candidate_id: str | None
    champion_id: str | None
    feedback_status: str
    lifecycle_status: str
    decision: str | None
    admitted: bool
    transitioned: bool
    stable: bool
    replacement_allowed: bool
    rollback_allowed: bool
    reason: str


class ChampionImprovementCycle:
    """Coordinates performance feedback with the existing Champion lifecycle."""

    def __init__(self, lifecycle: Any, performance_feedback: Any) -> None:
        self.lifecycle = lifecycle
        self.performance_feedback = performance_feedback

    def run(
        self,
        candidate_id: str | None = None,
        champion_id: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        baseline: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> ChampionImprovementSnapshot:
        metrics = dict(metrics or {})
        baseline = dict(baseline or {})
        context = dict(context or {})

        feedback = self._call_feedback(metrics, baseline, context)

        lifecycle = self._call_lifecycle(
            candidate_id=candidate_id,
            champion_id=champion_id,
            metrics=metrics,
            baseline=baseline,
            feedback=feedback,
            context=context,
        )

        return self._build_snapshot(
            candidate_id=candidate_id,
            champion_id=champion_id,
            feedback=feedback,
            lifecycle=lifecycle,
        )

    def evaluate(
        self,
        candidate_id: str | None = None,
        champion_id: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        baseline: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> ChampionImprovementSnapshot:
        return self.run(
            candidate_id=candidate_id,
            champion_id=champion_id,
            metrics=metrics,
            baseline=baseline,
            context=context,
        )

    def _call_feedback(
        self,
        metrics: Mapping[str, Any],
        baseline: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Any:
        method = self._find_method(
            self.performance_feedback,
            ("analyze", "evaluate", "assess", "run"),
        )

        if method is None:
            return {}

        return self._invoke(
            method,
            metrics=metrics,
            baseline=baseline,
            context=context,
        )

    def _call_lifecycle(
        self,
        candidate_id: str | None,
        champion_id: str | None,
        metrics: Mapping[str, Any],
        baseline: Mapping[str, Any],
        feedback: Any,
        context: Mapping[str, Any],
    ) -> Any:
        method = self._find_method(
            self.lifecycle,
            ("run", "evaluate"),
        )

        if method is None:
            return {}

        return self._invoke(
            method,
            candidate_id=candidate_id,
            champion_id=champion_id,
            metrics=metrics,
            baseline=baseline,
            feedback=feedback,
            context=context,
        )

    @staticmethod
    def _find_method(target: Any, names: tuple[str, ...]) -> Any:
        for name in names:
            method = getattr(target, name, None)

            if callable(method):
                return method

        return None

    @staticmethod
    def _invoke(method: Any, **kwargs: Any) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(**kwargs)

        parameters = signature.parameters

        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

        if accepts_kwargs:
            return method(**kwargs)

        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in parameters
        }

        return method(**filtered)

    def _build_snapshot(
        self,
        candidate_id: str | None,
        champion_id: str | None,
        feedback: Any,
        lifecycle: Any,
    ) -> ChampionImprovementSnapshot:
        feedback_status = self._read_status(
            feedback,
            default="UNKNOWN",
        )

        lifecycle_status = self._read_status(
            lifecycle,
            default="UNKNOWN",
        )

        decision = self._read_value(
            lifecycle,
            ("decision", "action"),
            default=None,
        )

        admitted = self._read_bool(
            lifecycle,
            "admitted",
        )

        transitioned = self._read_bool(
            lifecycle,
            "transitioned",
        )

        stable = self._read_bool(
            lifecycle,
            "stable",
        )

        replacement_allowed = self._read_bool(
            lifecycle,
            "replacement_allowed",
        )

        rollback_allowed = self._read_bool(
            lifecycle,
            "rollback_allowed",
        )

        reason = self._read_value(
            lifecycle,
            ("reason",),
            default=None,
        )

        if reason is None:
            reason = self._read_value(
                feedback,
                ("reason", "diagnosis"),
                default="",
            )

        status = self._resolve_status(
            feedback_status=feedback_status,
            lifecycle_status=lifecycle_status,
            admitted=admitted,
            transitioned=transitioned,
            stable=stable,
            replacement_allowed=replacement_allowed,
            rollback_allowed=rollback_allowed,
        )

        return ChampionImprovementSnapshot(
            status=status,
            candidate_id=candidate_id,
            champion_id=champion_id,
            feedback_status=feedback_status,
            lifecycle_status=lifecycle_status,
            decision=decision,
            admitted=admitted,
            transitioned=transitioned,
            stable=stable,
            replacement_allowed=replacement_allowed,
            rollback_allowed=rollback_allowed,
            reason=str(reason or ""),
        )

    @staticmethod
    def _resolve_status(
        feedback_status: str,
        lifecycle_status: str,
        admitted: bool,
        transitioned: bool,
        stable: bool,
        replacement_allowed: bool,
        rollback_allowed: bool,
    ) -> str:
        if rollback_allowed:
            return "ROLLBACK"

        if transitioned and stable:
            return "PROMOTED"

        if replacement_allowed and admitted:
            return "READY"

        if admitted:
            return "ADMITTED"

        if feedback_status in {"WARNING", "DEGRADED"}:
            return feedback_status

        if lifecycle_status in {"WARNING", "DEGRADED"}:
            return lifecycle_status

        return "EVALUATED"

    @staticmethod
    def _read_status(
        value: Any,
        default: str,
    ) -> str:
        status = ChampionImprovementCycle._read_value(
            value,
            ("status",),
            default=default,
        )

        return str(status)

    @staticmethod
    def _read_value(
        value: Any,
        names: tuple[str, ...],
        default: Any,
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
    def _read_bool(
        value: Any,
        name: str,
    ) -> bool:
        result = ChampionImprovementCycle._read_value(
            value,
            (name,),
            default=False,
        )

        return bool(result)