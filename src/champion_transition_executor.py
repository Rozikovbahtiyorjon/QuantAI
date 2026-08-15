from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ChampionTransitionResult:
    action: str
    changed: bool
    champion: dict[str, Any]
    reason: str


class ChampionTransitionExecutor:
    VALID_ACTIONS = {
        "PROMOTE",
        "REPLACE",
        "KEEP",
        "HOLD",
        "REJECT",
    }

    def execute(
        self,
        decision: Any,
        current_champion: Mapping[str, Any] | None = None,
        candidate: Mapping[str, Any] | None = None,
    ) -> ChampionTransitionResult:
        action = str(getattr(decision, "action", "")).upper()
        reason = str(getattr(decision, "reason", ""))

        current = dict(current_champion or {})
        proposed = dict(candidate or {})

        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Unsupported transition action: {action}")

        if action == "PROMOTE":
            if not proposed:
                return ChampionTransitionResult(
                    action="REJECT",
                    changed=False,
                    champion=current,
                    reason="candidate_missing",
                )

            return ChampionTransitionResult(
                action="PROMOTE",
                changed=True,
                champion=proposed,
                reason=reason or "candidate_promoted",
            )

        if action == "REPLACE":
            if not proposed:
                return ChampionTransitionResult(
                    action="REJECT",
                    changed=False,
                    champion=current,
                    reason="candidate_missing",
                )

            return ChampionTransitionResult(
                action="REPLACE",
                changed=True,
                champion=proposed,
                reason=reason or "champion_replaced",
            )

        if action == "KEEP":
            return ChampionTransitionResult(
                action="KEEP",
                changed=False,
                champion=current,
                reason=reason or "champion_kept",
            )

        if action == "HOLD":
            return ChampionTransitionResult(
                action="HOLD",
                changed=False,
                champion=current,
                reason=reason or "transition_held",
            )

        return ChampionTransitionResult(
            action="REJECT",
            changed=False,
            champion=current,
            reason=reason or "candidate_rejected",
        )