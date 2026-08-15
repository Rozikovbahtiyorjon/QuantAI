from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ChampionLifecycleSnapshot:
    status: str
    candidate_id: str
    champion_id: str
    admitted: bool
    decision: str
    transitioned: bool
    stable: bool
    replacement_allowed: bool
    rollback_allowed: bool
    reason: str = ""


class ChampionLifecycle:
    """
    Thin orchestration layer for the existing Champion lifecycle components.

    The lifecycle does not create a new trading decision. It coordinates:
    evaluation -> admission -> transition decision -> transition execution
    -> stability -> replacement/rollback guards.
    """

    def __init__(
        self,
        evaluator: Any,
        admission_controller: Any,
        transition_decision: Any,
        transition_executor: Any,
        stability_monitor: Any,
        replacement_guard: Any,
        rollback_guard: Any,
    ) -> None:
        self.evaluator = evaluator
        self.admission_controller = admission_controller
        self.transition_decision = transition_decision
        self.transition_executor = transition_executor
        self.stability_monitor = stability_monitor
        self.replacement_guard = replacement_guard
        self.rollback_guard = rollback_guard

    @staticmethod
    def _call(
        component: Any,
        method_names: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        for name in method_names:
            method = getattr(component, name, None)
            if callable(method):
                try:
                    return method(*args, **kwargs)
                except TypeError:
                    if kwargs:
                        try:
                            return method(*args)
                        except TypeError:
                            continue
                    continue

        raise AttributeError(
            f"{type(component).__name__} does not expose any of: "
            + ", ".join(method_names)
        )

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

        for name in names:
            if hasattr(value, name):
                return getattr(value, name)

        return default

    @staticmethod
    def _bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default

        return bool(value)

    def evaluate(
        self,
        candidate_metrics: Mapping[str, float],
        champion_metrics: Mapping[str, float],
    ) -> Any:
        return self._call(
            self.evaluator,
            ("evaluate", "compare"),
            candidate_metrics,
            champion_metrics,
        )

    def run(
        self,
        candidate_id: str,
        champion_id: str,
        candidate_metrics: Mapping[str, float],
        champion_metrics: Mapping[str, float],
        context: Mapping[str, Any] | None = None,
    ) -> ChampionLifecycleSnapshot:
        if not candidate_id:
            raise ValueError("candidate_id must not be empty.")

        if not champion_id:
            raise ValueError("champion_id must not be empty.")

        context = dict(context or {})

        evaluation = self.evaluate(
            candidate_metrics,
            champion_metrics,
        )

        qualified = self._bool(
            self._read(
                evaluation,
                "qualified",
                "accepted",
                "is_qualified",
                default=False,
            )
        )

        if not qualified:
            return ChampionLifecycleSnapshot(
                status="REJECTED",
                candidate_id=candidate_id,
                champion_id=champion_id,
                admitted=False,
                decision="REJECT",
                transitioned=False,
                stable=True,
                replacement_allowed=False,
                rollback_allowed=False,
                reason="candidate evaluation did not qualify",
            )

        admission = self._call(
            self.admission_controller,
            ("admit", "evaluate", "check", "allow"),
            evaluation,
            context,
        )

        admitted = self._bool(
            self._read(
                admission,
                "admitted",
                "accepted",
                "allowed",
                "approved",
                default=admission,
            )
        )

        if not admitted:
            return ChampionLifecycleSnapshot(
                status="REJECTED",
                candidate_id=candidate_id,
                champion_id=champion_id,
                admitted=False,
                decision="REJECT",
                transitioned=False,
                stable=True,
                replacement_allowed=False,
                rollback_allowed=False,
                reason=self._read(
                    admission,
                    "reason",
                    "message",
                    default="admission denied",
                ),
            )

        decision = self._call(
            self.transition_decision,
            ("decide", "evaluate", "check"),
            candidate_id,
            champion_id,
            evaluation,
            context,
        )

        decision_value = str(
            self._read(
                decision,
                "decision",
                "action",
                "status",
                default=decision,
            )
        ).upper()

        if decision_value not in {
            "APPROVE",
            "ACCEPT",
            "PROMOTE",
            "TRANSITION",
            "REPLACE",
        }:
            return ChampionLifecycleSnapshot(
                status="REJECTED",
                candidate_id=candidate_id,
                champion_id=champion_id,
                admitted=True,
                decision=decision_value,
                transitioned=False,
                stable=True,
                replacement_allowed=False,
                rollback_allowed=False,
                reason=self._read(
                    decision,
                    "reason",
                    "message",
                    default="transition decision did not approve replacement",
                ),
            )

        transition = self._call(
            self.transition_executor,
            ("execute", "transition", "promote", "apply"),
            candidate_id,
            champion_id,
            context,
        )

        transitioned = self._bool(
            self._read(
                transition,
                "transitioned",
                "executed",
                "success",
                "promoted",
                default=transition,
            )
        )

        stability = self._call(
            self.stability_monitor,
            ("analyze", "check", "evaluate"),
            context,
        )

        stable = self._bool(
            self._read(
                stability,
                "stable",
                "is_stable",
                default=False,
            )
        )

        replacement = self._call(
            self.replacement_guard,
            ("allow", "check", "evaluate"),
            evaluation,
            stability,
            context,
        )

        replacement_allowed = self._bool(
            self._read(
                replacement,
                "allowed",
                "approved",
                "replacement_allowed",
                default=replacement,
            )
        )

        rollback = self._call(
            self.rollback_guard,
            ("allow", "check", "evaluate"),
            stability,
            context,
        )

        rollback_allowed = self._bool(
            self._read(
                rollback,
                "allowed",
                "approved",
                "rollback_allowed",
                default=rollback,
            )
        )

        if not transitioned:
            status = "REJECTED"
            reason = "transition executor did not complete transition"
        elif not stable:
            status = "DEGRADED"
            reason = "transition completed but champion is not stable"
        else:
            status = "PROMOTED"
            reason = "candidate completed the controlled lifecycle"

        return ChampionLifecycleSnapshot(
            status=status,
            candidate_id=candidate_id,
            champion_id=champion_id,
            admitted=True,
            decision=decision_value,
            transitioned=transitioned,
            stable=stable,
            replacement_allowed=replacement_allowed,
            rollback_allowed=rollback_allowed,
            reason=reason,
        )


__all__ = [
    "ChampionLifecycle",
    "ChampionLifecycleSnapshot",
]